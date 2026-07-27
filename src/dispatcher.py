"""Fair dispatcher — the WFQ scheduler that sits in front of Prefect.

Why this exists: if the API enqueued every video/document to Prefect at
register-time, Prefect would run them in submitted order (FIFO) — one user
who uploads 50 videos (or backfills 50 papers) blocks everyone behind them.
Instead, both wait `pending` in Postgres and THIS loop admits them:

  every DISPATCH_INTERVAL_S:
    slots = DISPATCH_MAX_INFLIGHT - (videos+documents currently queued/running)
    claim up to `slots` pending rows in FAIR order (round-robin across users,
      videos and documents pooled together — see db.wfq_claim_all)
    schedule the right Prefect deployment for each

Because only ~capacity items are ever handed to Prefect at once, the *waiting
line lives in our DB, fairly ordered* rather than FIFO inside Prefect. No user
— and no source TYPE — can starve the others. Set ENABLE_FAIR_DISPATCH=false
to fall back to immediate FIFO enqueue (useful for A/B teaching the difference).

Runs as a background thread in worker.py. With one worker that's exact; with
several, each runs a dispatcher — the atomic claim keeps rows handed out once,
at worst mildly over-admitting (harmless; Prefect still caps execution).
"""
from __future__ import annotations

import threading
import time

from . import config, db, jobs

_ENQUEUE = {"video": jobs.enqueue_video, "document": jobs.enqueue_document}
_RESET_STATUS = {"video": db.set_status, "document": db.set_doc_status}


def dispatch_once() -> int:
    """Admit as many fairly-chosen pending videos/documents as free capacity
    allows. Returns how many were dispatched this tick."""
    slots = config.DISPATCH_MAX_INFLIGHT - db.count_inflight()
    if slots <= 0:
        return 0
    claimed = db.wfq_claim_all(slots)
    for row in claimed:
        try:
            _ENQUEUE[row["entity"]](row["id"], row["user_id"])
        except Exception as exc:
            # Couldn't reach Prefect — put it back so it's retried next tick.
            _RESET_STATUS[row["entity"]](row["id"], "pending", error=f"dispatch: {exc}")
    if claimed:
        print(f"[dispatch] admitted {len(claimed)} item(s) "
              f"({db.count_inflight()}/{config.DISPATCH_MAX_INFLIGHT} in flight)")
    return len(claimed)


def run_forever() -> None:
    print(f"[dispatch] fair scheduler on — max in-flight "
          f"{config.DISPATCH_MAX_INFLIGHT}, tick {config.DISPATCH_INTERVAL_S}s")
    while True:
        try:
            dispatch_once()
        except Exception as exc:  # never let the scheduler thread die
            print(f"[dispatch] error: {type(exc).__name__}: {exc}")
        time.sleep(config.DISPATCH_INTERVAL_S)


def start_in_background() -> None:
    """Start the dispatcher as a daemon thread (no-op if fair dispatch is off)."""
    if not config.ENABLE_FAIR_DISPATCH:
        print("[dispatch] fair dispatch disabled — FIFO (immediate enqueue)")
        return
    threading.Thread(target=run_forever, daemon=True, name="dispatcher").start()
