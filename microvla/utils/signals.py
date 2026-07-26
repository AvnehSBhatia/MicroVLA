"""Decline SIGTERM, because the training host reaps jobs from outside.

Measured on the MI300X box: processes die with exit 143 (128+15 = SIGTERM) at
unpredictable times — a 60 GB allocation succeeds, a 500 MB one dies, the same
command alternates between working and failing. `dmesg` is empty, no OOM daemon
is visible, RAM and VRAM are both mostly free. Whatever is killing jobs lives on
the host, outside the container's PID namespace, so it cannot be found or
configured from inside.

SIGTERM means "please stop" and IS catchable, so a long-running job can decline
it. Every CLI in this repo installs this at startup, because every one of them
is long enough to be reaped: a bake is ~10 min, stage A ~1 h, a closed-loop eval
~25 min.

What this does NOT do: SIGKILL cannot be caught. If a process still dies, its
exit code is 137 rather than 143, and the reaper escalates — the defence there is
`train_batched.py --resume-stage-a`, which banks progress every epoch.

Consequences worth knowing:

* ``kill <pid>`` no longer stops these processes. Use ``kill -9 <pid>``.
* Ctrl-C is SIGINT and unaffected, so interactive interruption works as before.
* ``MICROVLA_ALLOW_SIGTERM=1`` restores default behaviour, e.g. on a machine
  where a supervisor legitimately needs to stop the job, or where the reaper is
  enforcing a fair-use policy you should not be overriding.
"""

from __future__ import annotations

import os
import signal

#: Set to 1 to opt out and let SIGTERM terminate normally.
ENV_OPT_OUT = "MICROVLA_ALLOW_SIGTERM"

_installed = False


def ignore_sigterm(verbose: bool = True) -> bool:
    """Installs SIG_IGN for SIGTERM unless opted out. Idempotent.

    Args:
        verbose: Print a one-line notice the first time it takes effect, so a
            non-responsive ``kill`` is never a mystery.

    Returns:
        True if SIGTERM is now ignored, False if opted out or unavailable.
    """
    global _installed
    if os.environ.get(ENV_OPT_OUT, "").strip() not in ("", "0", "false", "False"):
        return False
    if _installed:
        return True
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (ValueError, OSError, AttributeError):
        # Non-main thread, or a platform without SIGTERM. Not worth failing over.
        return False
    _installed = True
    if verbose:
        print(f"[signals] SIGTERM ignored (host reaps jobs; see "
              f"microvla/utils/signals.py). `kill <pid>` will not work — use "
              f"`kill -9`. Opt out with {ENV_OPT_OUT}=1.", flush=True)
    return True
