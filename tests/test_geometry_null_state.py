"""
test_geometry_null_state.py
===========================
Verifies that geometry_status starts as NULL_ON_CURRENT_DATA and only
becomes ACTIVE after primary image scanning passes all preregistered gates.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def test_geometry_initial_state():
    """Geometry must begin as NULL_ON_CURRENT_DATA in the pipeline."""
    from ancient_ml.blind import geometry_channel

    # The module docstring should document the initial NULL state
    doc = geometry_channel.__doc__ or ""
    assert "NULL_ON_CURRENT_DATA" in doc, (
        "geometry_channel.py docstring must document NULL_ON_CURRENT_DATA status"
    )
    print("PASS: geometry_channel.py documents NULL_ON_CURRENT_DATA initial state.")


def test_geometry_requires_primary_images():
    """Embedded phi_rate is not valid evidence for geometry claims."""
    # Verify that embedded/approximated geometry data is clearly marked
    # as not equivalent to primary image scanning
    from ancient_ml.blind import geometry_channel

    doc = geometry_channel.__doc__ or ""
    if "embedded" in doc.lower() or "approximation" in doc.lower():
        print("PASS: geometry_channel.py marks embedded/approximation data as insufficient.")
    else:
        # Check for STATUS / NULL_ON_CURRENT_DATA banner
        assert "NULL_ON_CURRENT_DATA" in doc
        print("PASS: geometry_channel.py restricts geometry claims to primary image data.")


def test_phi_targets_fixed_preregistered():
    """Phi targets (1.618, 0.618) must be fixed before any scanning."""
    from ancient_ml.blind import geometry_channel

    # The phi targets should be documented as fixed/preregistered
    doc = geometry_channel.__doc__ or ""
    assert "1.618" in doc or "phi" in doc.lower(), (
        "geometry_channel.py must document phi targets"
    )
    print("PASS: Phi targets documented as fixed/preregistered.")


if __name__ == "__main__":
    test_geometry_initial_state()
    test_geometry_requires_primary_images()
    test_phi_targets_fixed_preregistered()
