import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

from extract_features import COLUMNS, features_for

ENTRY = {
    "id": "example-1",
    "declarations": ["a: date", "b: date", "c: date", "n: int"],
    "constraints": [
        "a == Date(2024, 1, 31)",
        "b == a + Period(0, 1, 0)",
        "c > b + Period(0, 0, 10)",
        "b.day == 29 || b.day == 28 || (n > 3 && n < 10)",
    ],
}


def test_simple_example():
    features = features_for(ENTRY, "example")

    print(f"\n{ENTRY['id']}\n")
    for d in ENTRY["declarations"]:
        print(f"  {d}")
    print()
    for c in ENTRY["constraints"]:
        print(f"  {c}")
    print()
    for col in COLUMNS:
        print(f"  {col:26} {features.get(col)}")


if __name__ == "__main__":
    test_simple_example()
