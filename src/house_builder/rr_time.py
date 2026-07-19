"""Shared monotonic time source for Rerun logging across house_builder modules.

The build lifecycle is event-driven (state transitions, policy calls, verification
checks) rather than a fixed-rate loop, so there's no natural per-tick index to log
against. This gives every module in the package a single shared "harness_step"
timeline instead, so events from different modules interleave in the right order.
"""
import itertools

import rerun as rr

_counter = itertools.count()


def log_step() -> None:
    """Advances the shared harness_step timeline and sets it as the current time."""
    rr.set_time("harness_step", sequence=next(_counter))
