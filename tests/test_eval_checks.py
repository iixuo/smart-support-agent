import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import check_case, normalize  # noqa: E402

from support_agent import db


def test_normalize_numbers_and_ranges():
    assert "1-3" in normalize("1～3 个工作日")
    assert "1-3" in normalize("1 至 3 天")
    assert "1299" in normalize("¥1,299.00")


def test_check_amounts_and_ticket():
    conn = db.init_db(":memory:")
    case = {"checks": {"amounts": [552.78], "ticket": False, "tools": ["calculate_refund"]}}
    assert check_case(case, "可退 552.78 元", ["calculate_refund"], conn, True) == []
    assert check_case(case, "可退 553 元", ["calculate_refund"], conn, True) == ["missing amount 552.78"]
    assert check_case(case, "可退 552.78 元", [], conn, False) == []  # baselines are not judged on tools
