"""Load the team's fixed-holdout Exp1 module despite its parenthesized filename."""

import hashlib
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "data_preparation(exp1)v2.py"
spec = importlib.util.spec_from_file_location("_hw2_exp1_v2", SOURCE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def source_metadata():
    return {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "preparation_protocol": "fixed Table 5 holdouts before train-only IQR/sampling",
    }
