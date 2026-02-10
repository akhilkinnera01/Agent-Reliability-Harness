"""
Build the groundedness benchmark by sampling the HaluEval QA split verbatim.

No questions or answers are authored here. Each HaluEval record already ships a
source (knowledge), a correct answer, and a human-verified hallucinated answer.
We take the first N records and split each into two labeled cases, so the sample
is deterministic and reproducible.

    python -m benchmarks.build_dataset            # 50 records  -> 100 cases
    python -m benchmarks.build_dataset 250        # 250 records -> 500 cases

Source: https://github.com/RUCAIBox/HaluEval (MIT). Paper: arXiv:2305.11747.
"""

import json
import pathlib
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
OUT = pathlib.Path(__file__).parent / "datasets" / "groundedness_cases.json"


def build(n_records: int) -> dict:
    raw = urllib.request.urlopen(URL, timeout=120).read().decode()
    records = [json.loads(l) for l in raw.splitlines() if l.strip()][:n_records]
    cases = []
    for r in records:
        cases.append({"grounded": True,  "source": r["knowledge"], "answer": r["right_answer"]})
        cases.append({"grounded": False, "source": r["knowledge"], "answer": r["hallucinated_answer"]})
    return {
        "description": (
            "HaluEval QA split, sampled verbatim (no authored content). "
            f"First {n_records} records -> {len(cases)} balanced cases; "
            "knowledge=source, right_answer=grounded, hallucinated_answer=not grounded."
        ),
        "source_url": "https://github.com/RUCAIBox/HaluEval",
        "n": len(cases),
        "cases": cases,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    OUT.write_text(json.dumps(build(n), indent=2) + "\n")
    print(f"wrote {OUT}  ({2 * n} cases from {n} HaluEval records)")
