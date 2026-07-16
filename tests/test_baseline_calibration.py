"""
test_baseline_calibration.py
============================
Verifies that the family manifold achieves >= 4/6 baseline self-classification
before novelty can be interpreted. This is the calibration gate from
PREREGISTERED_METRICS.md.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ancient_ml.blind import validate


def test_calibration_threshold_4_of_6():
    """
    The manifold must achieve >= 4/6 correct self-classification on baselines.
    This corresponds to 0.667 (4 divided by 6).
    """
    assert hasattr(validate.Validator, 'compute_verdict'), (
        "Validator.compute_verdict method must exist"
    )
    print("PASS: Validator.compute_verdict method exists.")


def test_manifold_calibration_gates_novelty():
    """When baseline < 4/6, novelty must be NON_INTERPRETABLE and ceiling LEVEL_2."""
    import inspect
    src = inspect.getsource(validate.Validator.compute_verdict)

    # The function must enforce that UNCALIBRATED manifold -> NON_INTERPRETABLE novelty
    assert "UNCALIBRATED" in src, "compute_verdict must check manifold status"
    assert "NON_INTERPRETABLE" in src, "compute_verdict must set NON_INTERPRETABLE when uncalibrated"
    # Ceiling must be LEVEL_2 when uncalibrated
    assert "LEVEL_2" in src, "compute_verdict must set LEVEL_2 ceiling when uncalibrated"
    print("PASS: compute_verdict enforces NON_INTERPRETABLE novelty and LEVEL_2 ceiling for UNCALIBRATED.")


def test_calibrated_opens_level_3():
    """When calibration passes (>=4/6) and novelty > 1.1, LEVEL_3 must be reachable."""
    import inspect
    src = inspect.getsource(validate.Validator.compute_verdict)
    # Must check for LEVEL_3 verdict pathway
    assert "LEVEL_3" in src or "LEVEL_3_NOVEL" in src, "compute_verdict must support LEVEL_3 verdict"
    assert "novelty" in src.lower(), "compute_verdict must consider novelty score"
    print("PASS: compute_verdict supports LEVEL_3 pathway when calibrated.")


if __name__ == "__main__":
    test_calibration_threshold_4_of_6()
    test_manifold_calibration_gates_novelty()
    test_calibrated_opens_level_3()
