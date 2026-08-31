import json
import subprocess
import sys


def test_oft_package_import_is_lazy():
    """Catch the OFT front door eagerly importing runtime or CUDA modules."""
    code = """
import json
import sys
import sglang

names = (
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.oft.triton_ops",
    "torch.distributed",
)
before = {name: name in sys.modules for name in names}
import sglang.srt.oft
after = {name: name in sys.modules for name in names}
print(json.dumps({"before": before, "after": after}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    state = json.loads(result.stdout)
    assert state["after"]["sglang.srt.model_executor.model_runner"] is False
    assert state["after"]["sglang.srt.oft.triton_ops"] is False
    assert (
        state["after"]["torch.distributed"]
        == state["before"]["torch.distributed"]
    )
