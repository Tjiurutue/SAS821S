"""
08_live_capture_simulator.py
------------------------------
Standalone terminal demo: simulates events arriving one at a time (as if
from a real feed) and shows the ALREADY-TRAINED ML models capturing and
categorising each one live, with a short delay between events so it reads
like a real-time SOC feed.

Not connected to any real network or client - every event is synthetic,
generated from the same rule catalogue and ticket templates used to build
the training data, so the trained models score genuinely new, previously
unseen events (not ones they were trained on).

Usage:
    python 08_live_capture_simulator.py                  # 25 events, 1.5s apart
    python 08_live_capture_simulator.py --n 50 --interval 0.5
    python 08_live_capture_simulator.py --seed 42
"""

import argparse
import time
import numpy as np

from live_capture_lib import capture_one_event, append_to_log, load_models, LOG_PATH


def format_line(i, event):
    ts = event["captured_at"].split("T")[1]
    if event["event_type"] == "alert":
        return (f"[{ts}] #{i:03d}  ALERT   {event['client_id']}  "
                f"{event['description']:<48}  "
                f"tp_prob={event['tp_probability']}  {event['priority_tier']}")
    else:
        return (f"[{ts}] #{i:03d}  TICKET  {event['client_id']}  "
                f"category={event['predicted_category']:<14} "
                f"urgency={event['predicted_urgency']}")


def main():
    parser = argparse.ArgumentParser(description="Simulate a live event feed scored by the trained T13 models.")
    parser.add_argument("--n", type=int, default=25, help="number of events to capture (default 25)")
    parser.add_argument("--interval", type=float, default=1.5, help="seconds between events (default 1.5)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random each run)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    models = load_models()

    if models["clf"] is None:
        print("⚠️  No trained classifier found in 04_models/ — run 03_ml_models.py first.")
        print("    Continuing anyway; events will show as UNSCORED.\n")

    print(f"🔴 LIVE CAPTURE SIMULATION — {args.n} events, ~{args.interval}s apart")
    print(f"   (synthetic events, scored live by the trained supervised + text-mining models)")
    print(f"   Logging to: {LOG_PATH}\n")

    tier_counts = {"🔴 HIGH": 0, "🟡 MEDIUM": 0, "🟢 LOW": 0}
    ticket_count = 0

    try:
        for i in range(1, args.n + 1):
            event = capture_one_event(rng, models)
            print(format_line(i, event))
            append_to_log(event)

            if event["event_type"] == "alert":
                tier_counts[event["priority_tier"]] = tier_counts.get(event["priority_tier"], 0) + 1
            else:
                ticket_count += 1

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nStopped early by user (Ctrl+C).")

    print("\n" + "=" * 60)
    print("CAPTURE SUMMARY")
    print("=" * 60)
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count}")
    print(f"  🎫 Tickets categorised: {ticket_count}")
    print(f"\nFull log appended to: {LOG_PATH}")


if __name__ == "__main__":
    main()
