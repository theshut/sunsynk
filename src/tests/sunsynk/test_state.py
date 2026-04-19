"""Sunsynk sensor state."""

import logging

import pytest

from sunsynk.rwsensors import SystemTimeRWSensor
from sunsynk.sensors import BinarySensor, Sensor
from sunsynk.state import InverterState

_LOG = logging.getLogger(__name__)


def test_history(state: InverterState) -> None:
    """Test history with numeric values."""
    a = Sensor(1, "Some Value")
    state.track(a)

    state.update({1: 100})
    assert state[a] == 100
    assert state.history[a] == [100]

    state.update({1: 200})
    assert state[a] == 200
    assert state.history[a] == [100, 200]

    state.update({1: 300})
    assert state[a] == 300
    assert state.history[a] == [100, 200, 300]

    assert a not in state.historynn

    assert state.history_average(a) == 250
    assert state.history[a] == [250]


def test_history_raise(state: InverterState) -> None:
    """Test if we have a ValueError."""
    a = Sensor(2, "Some Value")
    state.track(a)
    with pytest.raises(ValueError):
        state.history_average(a)

    state.update({2: 100})
    assert state[a] == 100
    assert state.history_average(a) == 100
    assert state.history[a] == [100]

    state.update({2: 111})
    assert state.history[a] == [100, 111]
    assert state.history_average(a) == 111
    assert state.history[a] == [111]


def test_history_nn(state: InverterState) -> None:
    """Test history with non-numeric values."""
    a = SystemTimeRWSensor((1, 2, 3), "Some Value")
    state.track(a)

    state.update({1: 1, 2: 2, 3: 3})
    assert state.historynn[a] == [None, "2000-01-00 2:00:03"]
    assert a not in state.history

    state.update({1: 12, 2: 5, 3: 44})
    assert state.historynn[a] == ["2000-01-00 2:00:03", "2000-12-00 5:00:44"]
    assert a not in state.history


def test_history_nn_binary(state: InverterState) -> None:
    """Test history with non-numeric values."""
    a = BinarySensor((1), "Grid Connected")
    state.track(a)

    state.update({1: 1, 2: 2, 3: 3})
    assert state.historynn[a] == [None, True]
    assert a not in state.history

    state.update({1: 1, 2: 5, 3: 44})
    assert state.historynn[a] == [True, True]
    assert a not in state.history

    state.update({1: 0, 2: 5, 3: 44})
    assert state.historynn[a] == [True, False]
    assert a not in state.history


def test_zero_filter_ignores_sporadic_zeros(state: InverterState) -> None:
    """Test that sporadic zero values are ignored."""
    a = Sensor(1, "Power")
    state.track(a)

    state.update({1: 500})
    assert state[a] == 500

    # Single zero should be ignored
    state.update({1: 0})
    assert state[a] == 500

    # Second zero still ignored
    state.update({1: 0})
    assert state[a] == 500

    # Non-zero resets the counter
    state.update({1: 600})
    assert state[a] == 600


def test_zero_filter_accepts_after_threshold(state: InverterState) -> None:
    """Test that 3 consecutive zeros are accepted."""
    a = Sensor(1, "Power")
    state.track(a)

    state.update({1: 500})
    assert state[a] == 500

    state.update({1: 0})
    assert state[a] == 500

    state.update({1: 0})
    assert state[a] == 500

    # Third consecutive zero should be accepted
    state.update({1: 0})
    assert state[a] == 0


def test_zero_filter_resets_on_nonzero(state: InverterState) -> None:
    """Test that non-zero value resets the zero counter."""
    a = Sensor(1, "Power")
    state.track(a)

    state.update({1: 500})
    state.update({1: 0})  # count=1
    state.update({1: 0})  # count=2
    state.update({1: 300})  # resets count
    assert state[a] == 300

    # Must start counting again from scratch
    state.update({1: 0})  # count=1
    assert state[a] == 300
    state.update({1: 0})  # count=2
    assert state[a] == 300
    state.update({1: 0})  # count=3, accepted
    assert state[a] == 0


def test_zero_filter_disabled(state: InverterState) -> None:
    """Test that zero_filter=0 disables filtering."""
    state.zero_filter = 0
    a = Sensor(1, "Power")
    state.track(a)

    state.update({1: 500})
    assert state[a] == 500

    state.update({1: 0})
    assert state[a] == 0


def test_zero_filter_skips_binary_sensors(state: InverterState) -> None:
    """Test that BinarySensors are not filtered."""
    a = BinarySensor(1, "Grid Connected")
    state.track(a)

    state.update({1: 1})
    assert state[a] is True

    state.update({1: 0})
    assert state[a] is False


def test_zero_filter_first_read_none(state: InverterState) -> None:
    """Test that first read of zero on a new sensor is accepted (oldv is None)."""
    a = Sensor(1, "Power")
    state.track(a)
    assert state[a] is None

    # First read is zero - should be accepted since there's no previous value
    state.update({1: 0})
    assert state[a] == 0
