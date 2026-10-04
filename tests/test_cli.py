from steercert.cli import main


def test_demo_summary_and_check(tmp_path, capsys):
    out = tmp_path / "demo.json"
    assert main(["demo", "--n", "800", "--out", str(out)]) == 0
    assert out.exists()
    assert main(["summary", str(out)]) == 0
    assert main(["check", str(out), "--min-efficiency", "reference/gramian_top=0.99"]) == 0
    assert main(["check", str(out), "--min-efficiency", "probe=0.9"]) == 1
    assert main(["check", str(out), "--fail-on-warnings"]) == 1
    printed = capsys.readouterr().out
    assert "FAIL" in printed and "OK all thresholds met" in printed
