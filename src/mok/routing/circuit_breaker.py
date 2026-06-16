"""
src/mok/routing/circuit_breaker.py

Per-expert circuit breaker — P1.3.

States
------
CLOSED   — normal; requests flow through.
OPEN     — expert tripped; requests are rejected immediately.
HALF_OPEN — probe window; one request allowed to test recovery.

Transitions
-----------
CLOSED  → OPEN       : consecutive_failures >= THRESHOLD
OPEN    → HALF_OPEN  : reset_seconds elapsed since last failure
HALF_OPEN → CLOSED   : probe request succeeds
HALF_OPEN → OPEN     : probe request fails (resets timer)

Thread safety: each ExpertBreaker uses a threading.Lock.
The registry (ExpertCircuitBreakerRegistry) is also locked.

No external dependencies — pure stdlib.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Tunables (match configs/routing.json circuit_breaker section)
# ---------------------------------------------------------------------------

FAILURE_THRESHOLD: int = 3      # consecutive failures before OPEN
RESET_SECONDS: float = 60.0     # seconds in OPEN before → HALF_OPEN
HALF_OPEN_PROBE_COUNT: int = 1  # successful probes needed to CLOSE


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class BreakerState(Enum):
    CLOSED    = auto()
    OPEN      = auto()
    HALF_OPEN = auto()


# ---------------------------------------------------------------------------
# Per-expert breaker
# ---------------------------------------------------------------------------

@dataclass
class ExpertBreaker:
    """Circuit breaker for a single named expert."""

    expert_name: str
    failure_threshold: int = FAILURE_THRESHOLD
    reset_seconds: float = RESET_SECONDS
    half_open_probe_count: int = HALF_OPEN_PROBE_COUNT

    # internal state — not part of the public API
    _state: BreakerState = field(default=BreakerState.CLOSED, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _probe_successes: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._get_state_unlocked()

    def allow_request(self) -> bool:
        """Return True if a request should be forwarded to this expert."""
        with self._lock:
            state = self._get_state_unlocked()
            if state == BreakerState.CLOSED:
                return True
            if state == BreakerState.OPEN:
                return False
            # HALF_OPEN — allow exactly one probe at a time
            return True

    def record_success(self) -> None:
        """Call after the expert returns a successful response."""
        with self._lock:
            state = self._get_state_unlocked()
            if state == BreakerState.HALF_OPEN:
                self._probe_successes += 1
                if self._probe_successes >= self.half_open_probe_count:
                    self._reset()
            elif state == BreakerState.CLOSED:
                # Clear failure streak on any success
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Call after the expert raises an error."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()
            self._probe_successes = 0
            if self._consecutive_failures >= self.failure_threshold:
                self._state = BreakerState.OPEN

    def force_open(self) -> None:
        """Manually trip the breaker (e.g., expert unloaded)."""
        with self._lock:
            self._state = BreakerState.OPEN
            self._last_failure_time = time.monotonic()

    def force_close(self) -> None:
        """Manually reset the breaker (e.g., after operator intervention)."""
        with self._lock:
            self._reset()

    # ------------------------------------------------------------------
    # Internal helpers (must be called with _lock held)
    # ------------------------------------------------------------------

    def _get_state_unlocked(self) -> BreakerState:
        if self._state == BreakerState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.reset_seconds:
                self._state = BreakerState.HALF_OPEN
                self._probe_successes = 0
        return self._state

    def _reset(self) -> None:
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._probe_successes = 0
        self._last_failure_time = 0.0

    def status_dict(self) -> dict:
        """Snapshot for logging / health endpoints."""
        with self._lock:
            return {
                "expert": self.expert_name,
                "state": self._get_state_unlocked().name,
                "consecutive_failures": self._consecutive_failures,
                "seconds_since_last_failure": (
                    round(time.monotonic() - self._last_failure_time, 1)
                    if self._last_failure_time else None
                ),
            }


# ---------------------------------------------------------------------------
# Registry — one breaker per expert, lazy-created
# ---------------------------------------------------------------------------

class ExpertCircuitBreakerRegistry:
    """
    Process-level registry of per-expert circuit breakers.

    Usage
    -----
        registry = ExpertCircuitBreakerRegistry()

        # Before routing
        if not registry.allow(expert_name):
            raise RoutingError(f"Expert {expert_name!r} is circuit-broken.")

        try:
            result = backend.generate(expert, payload)
            registry.success(expert_name)
        except BackendInvocationError:
            registry.failure(expert_name)
            raise
    """

    def __init__(
        self,
        failure_threshold: int = FAILURE_THRESHOLD,
        reset_seconds: float = RESET_SECONDS,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._breakers: dict[str, ExpertBreaker] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, expert_name: str) -> ExpertBreaker:
        with self._lock:
            if expert_name not in self._breakers:
                self._breakers[expert_name] = ExpertBreaker(
                    expert_name=expert_name,
                    failure_threshold=self._threshold,
                    reset_seconds=self._reset_seconds,
                )
            return self._breakers[expert_name]

    def allow(self, expert_name: str) -> bool:
        return self._get_or_create(expert_name).allow_request()

    def success(self, expert_name: str) -> None:
        self._get_or_create(expert_name).record_success()

    def failure(self, expert_name: str) -> None:
        self._get_or_create(expert_name).record_failure()

    def trip(self, expert_name: str) -> None:
        """Manually open a breaker."""
        self._get_or_create(expert_name).force_open()

    def reset(self, expert_name: str) -> None:
        """Manually close a breaker."""
        self._get_or_create(expert_name).force_close()

    def state_of(self, expert_name: str) -> BreakerState:
        return self._get_or_create(expert_name).state

    def all_status(self) -> list[dict]:
        with self._lock:
            names = list(self._breakers.keys())
        return [self._get_or_create(n).status_dict() for n in names]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ExpertCircuitBreakerRegistry | None = None
_registry_lock = threading.Lock()


def get_circuit_breaker_registry() -> ExpertCircuitBreakerRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ExpertCircuitBreakerRegistry()
    return _registry
