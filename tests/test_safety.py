from src.safety.checker import SafetyChecker


def test_safety_blocks_dosage() -> None:
    checker = SafetyChecker()
    res = checker.check("You should take 500mg of ibuprofen after meals.")
    assert res.safe is False
    assert res.reason

