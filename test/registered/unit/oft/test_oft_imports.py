import json
import subprocess
import sys


def test_oft_package_import_is_lazy():
    """Catch the OFT front door eagerly importing runtime or CUDA modules."""
    code = """
import json
import sys
import sglang.srt.oft

names = (
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.oft.triton_ops",
    "torch.distributed",
)
print(json.dumps({name: name in sys.modules for name in names}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "sglang.srt.model_executor.model_runner": False,
        "sglang.srt.oft.triton_ops": False,
        "torch.distributed": False,
    }
