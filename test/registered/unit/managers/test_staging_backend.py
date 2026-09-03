# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestStagingBackendSelection(unittest.TestCase):
    def test_selects_oft_backend_once_staged_impl_chosen(self):
        from types import SimpleNamespace

        from sglang.srt.managers.staging_backend import get_staging_backend
        from sglang.srt.oft.staged_manager import OFTStagingBackend

        tm = SimpleNamespace(
            server_args=SimpleNamespace(enable_lora_staging=False, oft_impl="staged")
        )
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsInstance(get_staging_backend(tm, obj), OFTStagingBackend)

    def test_selects_no_backend_for_the_plain_sibling_impl(self):
        from types import SimpleNamespace

        from sglang.srt.managers.staging_backend import get_staging_backend

        tm = SimpleNamespace(
            server_args=SimpleNamespace(enable_lora_staging=False, oft_impl="sibling")
        )
        obj = SimpleNamespace(load_format="oft_adapter")
        self.assertIsNone(get_staging_backend(tm, obj))
