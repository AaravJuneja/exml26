"""Exact equivalence of the rollout hot path with the recorded fixture.

Observation histories, feature rows, scripted-bot actions and full `collect`
outputs must be bit-identical to those recorded before any optimisation
(see ctf/tests/equivalence.py; fixtures are per platform).
"""

import pytest

from ctf.tests import equivalence as E

EXPECTED = E.load().get(E.platform_key())
pytestmark = pytest.mark.skipif(
    EXPECTED is None, reason=f"no equivalence fixture for {E.platform_key()}")


def _first_bad(expected, actual):
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            return f"first mismatch in ticks {i * E.CHUNK}..{(i + 1) * E.CHUNK - 1}"
    return f"length {len(actual)} != {len(expected)}"


@pytest.fixture(scope="module")
def scripted():
    return E.run_scripted()


@pytest.fixture(scope="module")
def rollouts():
    return E.run_rollouts()


@pytest.mark.parametrize("stream", ["observations", "features", "actions", "events"])
def test_scripted_matches(scripted, stream):
    for name, expected in EXPECTED["scripted"].items():
        actual = scripted[name][stream]
        if stream == "events":
            assert actual == expected[stream], name
        else:
            assert actual == expected[stream], f"{name} {stream}: {_first_bad(expected[stream], actual)}"


@pytest.mark.parametrize("kind", sorted(E.ROLLOUTS))
def test_rollout_outputs(rollouts, kind):
    expected, actual = EXPECTED["rollout"][kind], rollouts[kind]
    assert sorted(actual) == sorted(expected)
    for key in sorted(expected):
        if key == "obs":
            assert actual[key] == expected[key], f"{kind} obs: {_first_bad(expected[key], actual[key])}"
        else:
            assert actual[key] == expected[key], f"{kind} {key}"
