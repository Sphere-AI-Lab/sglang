from types import SimpleNamespace

import pytest

from sglang.srt.oft.base.manager import AdapterManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "requested, resident_reloadable, pinned, expected",
    [
        ({None, "disk"}, False, False, False),
        ({None, "wire"}, False, False, True),
        ({"disk"}, False, False, True),
        ({None, "disk"}, True, False, True),
        ({None, "disk"}, True, True, False),
        ({None, "wire"}, True, True, True),
        ({None, "disk", "wire"}, True, False, False),
    ],
)
def test_admission_reserves_non_evictable_residents(
    requested, resident_reloadable, pinned, expected
):
    manager = AdapterManager.__new__(AdapterManager)
    manager.max_adapters_per_batch = 2
    manager.num_pinned = int(pinned)
    manager.adapters = {}
    manager.refs = {
        "wire": SimpleNamespace(pinned=pinned, reloadable=resident_reloadable),
        "disk": SimpleNamespace(pinned=False, reloadable=True),
        "nonresident-wire": SimpleNamespace(pinned=False, reloadable=False),
    }
    manager.memory_pool = SimpleNamespace(
        max_adapters_per_batch=2,
        uid_to_buffer_id={None: 0, "wire": 1},
    )
    assert manager.validate_batch(requested) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
