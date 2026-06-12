#!/usr/bin/env python3
"""Simple Reachy connection and body yaw test script.

Usage: python scripts/reachy_connect_check.py

This script attempts to instantiate ReachyMini with the default media backend,
falls back to media_backend='no_media' if necessary, and issues a small body_yaw
movement to verify control.

It prints clear diagnostics and exits.
"""
from __future__ import annotations
import argparse
import time
import traceback

try:
    from reachy_mini import ReachyMini
except Exception as e:
    print('❌ reachy_mini import failed:', e)
    raise


def _normalize_host(host: str) -> str:
    return "127.0.0.1" if host == "0.0.0.0" else host


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="reachy-mini.local")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    host = _normalize_host(args.host)
    connection_mode = (
        "localhost_only" if host in ("localhost", "127.0.0.1", "::1") else "network"
    )

    print(
        f'Attempting to create ReachyMini(context) with media_backend="default" '
        f"against {host}:{args.port} ({connection_mode})"
    )
    try:
        ctx = ReachyMini(
            media_backend="default",
            host=host,
            port=args.port,
            connection_mode=connection_mode,
        )
    except Exception as exc:
        print('⚠️ Default media backend failed:', exc)
        print('   ↪ Trying media_backend="no_media" for control-only mode')
        try:
            ctx = ReachyMini(
                media_backend="no_media",
                host=host,
                port=args.port,
                connection_mode=connection_mode,
            )
        except Exception as exc2:
            print('❌ reachy_mini init failed (no-media):', exc2)
            return 2

    print('Created ReachyMini context, entering...')
    try:
        r = ctx.__enter__()
    except Exception as exc:
        print('❌ Error entering Reachy context:', exc)
        traceback.print_exc()
        return 3

    try:
        print('✅ Reachy context entered. Printing summary:')
        try:
            print('Reachy:', r)
        except Exception:
            pass

        print('\nIssuing test body yaw +0.2 rad (0.6s)')
        try:
            r.goto_target(body_yaw=0.2, duration=0.6)
            print('Called goto_target(body_yaw=0.2)')
        except Exception as exc:
            print('⚠️ Error calling goto_target:', exc)
            traceback.print_exc()

        time.sleep(1.0)

        print('\nResetting body yaw to 0.0 rad (0.6s)')
        try:
            r.goto_target(body_yaw=0.0, duration=0.6)
            print('Called goto_target(body_yaw=0.0)')
        except Exception as exc:
            print('⚠️ Error resetting goto_target:', exc)
            traceback.print_exc()

        time.sleep(0.8)

    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass

    print('\nDone. If the robot moved, control is working.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
