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

    Inherits the unified adapter identity (adapter_id/adapter_name/adapter_path/
    adapter_version/pinned) from AdapterRef. The unique ``adapter_id`` eliminates
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

    def __init__(self, adapter_paths: Optional[List[OFTRef]] = None):
        assert adapter_paths is None or all(
            isinstance(oft, OFTRef) for oft in adapter_paths
        ), (
            "OFTRegistry's initial adapter refs must be OFTRef instances. "
            "Please file an issue if you see this error."
        )
        super().__init__(adapter_paths)

    async def get_version_by_id(
        self, adapter_id: Union[str, List[str], None]
    ) -> Union[int, List[Optional[int]], None]:
        """
        Return the current OFT version for an adapter ID.

        The tokenizer manager uses this to build radix-cache keys that
        distinguish KV produced by different on-policy OFT weights.
        """

        def _lookup(uid: Optional[str]) -> Optional[int]:
            if uid is None:
                return None
            for oft_ref in self._registry.values():
                if oft_ref.adapter_id == uid:
                    return oft_ref.adapter_version
            raise ValueError(f"OFT ID {uid} does not exist.")

        async with self._registry_lock.reader_lock:
            if isinstance(adapter_id, str) or adapter_id is None:
                return _lookup(adapter_id)
            if isinstance(adapter_id, list):
                return [_lookup(uid) for uid in adapter_id]
            raise TypeError("adapter_id must be None, a string, or a list of strings.")

    async def bump_version_by_id(self, adapter_id: str) -> OFTRef:
        """
        Increment the version for an already-registered adapter ID.

        This keeps the stable ``adapter_id`` used by the OFT memory pool while
        invalidating radix-cache keys after streamed on-policy updates.
        """

        async with self._registry_lock.writer_lock:
            for adapter_name, oft_ref in self._registry.items():
                if oft_ref.adapter_id == adapter_id:
                    new_ref = replace(
                        oft_ref, adapter_version=oft_ref.adapter_version + 1
                    )
                    self._registry[adapter_name] = new_ref
                    return new_ref
        raise ValueError(f"OFT ID {adapter_id} does not exist.")

    async def get_unregistered_ofts(self, adapter_name):
        return await self.get_unregistered_adapters(adapter_name)

    async def lru_oft_name(self, exclude_pinned=False):
        return await self.lru_adapter_name(exclude_pinned=exclude_pinned)

    @property
    def num_registered_ofts(self) -> int:
        return self.num_registered_adapters
