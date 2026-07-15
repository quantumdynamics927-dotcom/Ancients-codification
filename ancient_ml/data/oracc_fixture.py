"""
ORACC JSON Fixture (small sample for testing without downloading full corpus)
"""
import json
from pathlib import Path

FIXTURE_DATA = {
    "project": "etcsri",
    "texts": [
        {
            "id": "etcsri001",
            "type": "document",
            "nodes": [
                {
                    "type": "line",
                    "f": {
                        "cdl": [
                            {"f": "nam"},
                            {"f": "lu"},
                            {"f": "2"},
                        ]
                    }
                },
                {
                    "type": "line",
                    "f": {
                        "cdl": [
                            {"f": "gi"},
                            {"f": "gi"},
                            {"f": "na"},
                        ]
                    }
                },
                {
                    "type": "line",
                    "f": {
                        "cdl": [
                            {"f": "e2"},
                            {"f": "gal"},
                            {"f": "na"},
                        ]
                    }
                },
            ]
        },
        {
            "id": "etcsri002",
            "type": "document",
            "nodes": [
                {
                    "type": "line",
                    "f": {
                        "cdl": [
                            {"f": "lugal"},
                            {"f": "ki"},
                            {"f": "en"},
                        ]
                    }
                },
                {
                    "type": "line",
                    "f": {
                        "cdl": [
                            {"f": "nanna"},
                            {"f": "ki"},
                            {"f": "an"},
                        ]
                    }
                },
            ]
        },
    ]
}

# Write fixture to data/oracc directory
def install_fixture():
    """Install fixture JSON file for testing."""
    fixture_dir = Path(__file__).parent / "data" / "oracc"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "etcsri_sample.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(FIXTURE_DATA, f, ensure_ascii=False, indent=2)
    return fixture_path


if __name__ == "__main__":
    path = install_fixture()
    print(f"Fixture installed: {path}")
