"""
tests/test_circuit_breaker.py

Tests for mok.routing.circuit_breaker — per-expert circuit breaker (P1.3).
All timing is mocked via monkeypatching time.monotonic so tests run instantly.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mok.routing.circuit_breaker import (
    BreakerState,
    ExpertBreaker,
    ExpertCircuitBreakerRegistry,
    get_circuit_breaker_registry,
    FAILURE_THRESHOLD,
    RESET_SECONDS,
)


# ---------------------------------------------------------------------------
# ExpertBreaker — state transitions
# ---------------------------------------------------------------------------

def make_breaker(**kwargs) -> ExpertBreaker:
    defaults = dict(
        expert_name="test-expert",
        failure_threshold=FAILURE_THRESHOLD,
        reset_seconds=RESET_SECONDS,
    )
    defaults.update(kwargs)
    return ExpertBreaker(**defaults)


class TestExpertBreakerInitial:
    def test_starts_closed(self):
        b = make_breaker()
        assert b.state == BreakerState.CLOSED

    def test_allows_request_when_closed(self):
        b = make_breaker()
        assert b.allow_request() is True


class TestExpertBreakerTripping:
    def test_opens_after_threshold_failures(self):
        b = make_breaker(failure_threshold=3)
        for _ in range(3):
            b.record_failure()
        assert b.state == BreakerState.OPEN

    def test_does_not_open_before_threshold(self):
        b = make_breaker(failure_threshold=3)
        for _ in range(2):
            b.record_failure()
        assert b.state == BreakerState.CLOSED

    def test_blocks_requests_when_open(self):
        b = make_breaker(failure_threshold=1)
        b.record_failure()
        assert b.allow_request() is False

    def test_success_clears_failure_streak(self):
        b = make_breaker(failure_threshold=3)
        b.record_failure()
        b.record_failure()
        b.record_success()
        b.record_failure()  # streak reset — only 1 failure now
        assert b.state == BreakerState.CLOSED


class TestExpertBreakerHalfOpen:
    def test_transitions_to_half_open_after_reset_seconds(self):
        b = make_breaker(failure_threshold=1, reset_seconds=30.0)
        b.record_failure()
        assert b.state == BreakerState.OPEN

        # Simulate time passing beyond reset window
        with patch("mok.routing.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = b._last_failure_time + 31.0
            assert b.state == BreakerState.HALF_OPEN

    def test_allows_probe_when_half_open(self):
        b = make_breaker(failure_threshold=1, reset_seconds=30.0)
        b.record_failure()
        with patch("mok.routing.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = b._last_failure_time + 31.0
            assert b.allow_request() is True

    def test_closes_after_probe_success(self):
        b = make_breaker(failure_threshold=1, reset_seconds=30.0, half_open_probe_count=1)
        b.record_failure()
        with patch("mok.routing.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = b._last_failure_time + 31.0
            # Trigger HALF_OPEN transition
            _ = b.state
            b.record_success()
        assert b.state == BreakerState.CLOSED

    def test_reopens_on_probe_failure(self):
        b = make_breaker(failure_threshold=1, reset_seconds=30.0)
        b.record_failure()
        base_time = b._last_failure_time
        with patch("mok.routing.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = base_time + 31.0
            _ = b.state  # → HALF_OPEN
            mock_time.monotonic.return_value = base_time + 32.0
            b.record_failure()
            assert b._state == BreakerState.OPEN


class TestExpertBreakerManual:
    def test_force_open(self):
        b = make_breaker()
        b.force_open()
        assert b.allow_request() is False

    def test_force_close(self):
        b = make_breaker(failure_threshold=1)
        b.record_failure()
        b.force_close()
        assert b.state == BreakerState.CLOSED
        assert b.allow_request() is True

    def test_status_dict_keys(self):
        b = make_breaker()
        d = b.status_dict()
        assert "expert" in d
        assert "state" in d
        assert "consecutive_failures" in d
        assert d["expert"] == "test-expert"
        assert d["state"] == "CLOSED"


# ---------------------------------------------------------------------------
# ExpertCircuitBreakerRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def setup_method(self):
        self.reg = ExpertCircuitBreakerRegistry(failure_threshold=3, reset_seconds=60.0)

    def test_allows_unknown_expert_by_default(self):
        assert self.reg.allow("new-expert") is True

    def test_creates_breaker_on_first_access(self):
        self.reg.allow("alpha")
        assert self.reg.state_of("alpha") == BreakerState.CLOSED

    def test_trips_after_threshold_failures(self):
        for _ in range(3):
            self.reg.failure("beta")
        assert self.reg.state_of("beta") == BreakerState.OPEN
        assert self.reg.allow("beta") is False

    def test_success_prevents_trip(self):
        self.reg.failure("gamma")
        self.reg.failure("gamma")
        self.reg.success("gamma")
        self.reg.failure("gamma")  # streak reset
        assert self.reg.state_of("gamma") == BreakerState.CLOSED

    def test_manual_trip_and_reset(self):
        self.reg.trip("delta")
        assert self.reg.allow("delta") is False
        self.reg.reset("delta")
        assert self.reg.allow("delta") is True

    def test_independent_experts(self):
        for _ in range(3):
            self.reg.failure("expert-a")
        assert self.reg.allow("expert-a") is False
        assert self.reg.allow("expert-b") is True  # unaffected

    def test_all_status_returns_list(self):
        self.reg.allow("x")
        self.reg.allow("y")
        statuses = self.reg.all_status()
        assert isinstance(statuses, list)
        names = [s["expert"] for s in statuses]
        assert "x" in names
        assert "y" in names


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_singleton_returns_same_instance():
    a = get_circuit_breaker_registry()
    b = get_circuit_breaker_registry()
    assert a is b


def test_singleton_is_registry_type():
    assert isinstance(get_circuit_breaker_registry(), ExpertCircuitBreakerRegistry)
