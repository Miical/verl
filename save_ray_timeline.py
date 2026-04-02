#!/usr/bin/env python3
"""Save Ray timeline from a running Ray cluster at any time.

Usage:
    python save_ray_timeline.py                                  # default output
    python save_ray_timeline.py -o /tmp/my_timeline.json         # custom path
    python save_ray_timeline.py --ray-address 10.0.0.1:6379      # custom Ray head
    python save_ray_timeline.py --no-disconnect                  # keep Ray connection alive
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Save Ray timeline from a running Ray cluster.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path. Default: $WORKSPACE/logs/ray_timeline_<timestamp>.json",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray head address (default: RAY_ADDRESS env var, or 'auto').",
    )
    parser.add_argument(
        "--no-disconnect",
        action="store_true",
        help="Do not call ray.shutdown() after saving.",
    )
    return parser.parse_args()


def resolve_output_path(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).resolve()
    workspace = os.environ.get("WORKSPACE", "/workspace/verl_vla")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(workspace) / "logs" / f"ray_timeline_{timestamp}.json"


def main():
    args = parse_args()

    os.environ.setdefault("RAY_grpc_max_message_size", "2147483647")

    import ray

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ray_address = args.ray_address or os.environ.get("RAY_ADDRESS", "auto")

    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
    except Exception as e:
        print(f"ERROR: Cannot connect to Ray cluster ({ray_address}): {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ray.timeline(filename=str(output_path))
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Ray timeline saved: {output_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"ERROR: Failed to save timeline: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not args.no_disconnect:
            ray.shutdown()


if __name__ == "__main__":
    main()
