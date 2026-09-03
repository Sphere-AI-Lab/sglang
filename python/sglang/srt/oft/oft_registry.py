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


from dataclasses import dataclass, fields, replace
from typing import List, Optional, Union

from sglang.srt.oft.base.registry import AdapterRef, AdapterRegistry


@dataclass(frozen=True)
class OFTRef(AdapterRef):
    """
    Reference record for an OFT adapter.

    Inherits the unified adapter identity (oft_id/oft_name/oft_path/
    version/pinned) from AdapterRef. The unique ``oft_id`` eliminates
    conflicts from reused names or paths and can be used to generate deterministic
    cache keys (e.g., radix cache).
    """

    def __str__(self) -> str:
        parts = [
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class OFTRegistry(AdapterRegistry):
    """
    The central registry to keep track of available OFT adapters and ongoing OFT requests.

    The `OFTRegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available OFT adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    """

    def __init__(self, oft_paths: Optional[List[OFTRef]] = None):
        assert oft_paths is None or all(
            isinstance(oft, OFTRef) for oft in oft_paths
        ), (
            "OFTRegistry's initial adapter refs must be OFTRef instances. "
            "Please file an issue if you see this error."
        )
        super().__init__(oft_paths)

    async def bump_version_by_id(self, oft_id: str) -> OFTRef:
        """
        Increment the version for an already-registered adapter ID.

        This keeps the stable ``oft_id`` used by the OFT memory pool while
        invalidating radix-cache keys after streamed on-policy updates.
        """

        async with self._registry_lock.writer_lock:
            for oft_name, oft_ref in self._registry.items():
                if oft_ref.oft_id == oft_id:
                    new_ref = replace(
                        oft_ref, version=oft_ref.version + 1
                    )
                    self._registry[oft_name] = new_ref
                    return new_ref
        raise ValueError(f"OFT ID {oft_id} does not exist.")

    async def lru_oft_name(self, exclude_pinned=False):
        return await self.lru_adapter_name(exclude_pinned=exclude_pinned)

    @property
    def num_registered_ofts(self) -> int:
        return self.num_registered_adapters
