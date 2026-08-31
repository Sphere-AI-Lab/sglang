"""Permanent guard against restoring the retired legacy PEFT surface."""

import re
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
HISTORICAL_DOC_DIRS = {
    REPO_ROOT / "docs" / "superpowers" / "plans",
    REPO_ROOT / "docs" / "superpowers" / "specs",
}
TEXT_SUFFIXES = {
    ".html",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
FORBIDDEN = {
    "legacy import": re.compile(r"sglang\.srt\.peft"),
    "legacy package path": re.compile(r"python/sglang/srt/peft"),
    "legacy implementation selector": re.compile(r"\boft_impl\b|--oft-impl\b"),
    "legacy LoRA CLI selector": re.compile(
        r"--peft-method(?:\s+|[\"'],\s*[\"'])lora\b", re.IGNORECASE
    ),
    "legacy LoRA runtime branch": re.compile(
        r"\bpeft_method\s*(?:==|=)\s*[\"']lora[\"']", re.IGNORECASE
    ),
}


def _is_historical(path: Path) -> bool:
    return any(root == path or root in path.parents for root in HISTORICAL_DOC_DIRS)


def _scanned_files():
    this_file = Path(__file__).resolve()
    for root_name in ("python", "test", "docs"):
        for path in (REPO_ROOT / root_name).rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and path.resolve() != this_file
                and not _is_historical(path)
            ):
                yield path


def test_legacy_peft_package_and_selectors_are_absent():
    assert not (REPO_ROOT / "python" / "sglang" / "srt" / "peft").exists()

    violations = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line}: {label}")

    assert not violations, "Retired PEFT surface reintroduced:\n" + "\n".join(violations)
