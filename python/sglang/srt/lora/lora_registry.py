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


import asyncio
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from uuid import NAMESPACE_URL, uuid4, uuid5

import msgspec
from msgspec.structs import fields, replace

from sglang.srt.utils import ConcurrentCounter
from sglang.srt.utils.aio_rwlock import RWLock


class LoRARef(msgspec.Struct, frozen=True, array_like=True):
    """
    Reference record for a LoRA model.

    This object guarantees a unique ``lora_id`` and may include ``lora_name``, ``lora_path``, and ``pinned``.
    The ID eliminates conflicts from reused LoRA names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """

    lora_id: str = msgspec.field(default_factory=lambda: uuid4().hex)
    lora_name: Optional[str] = None
    lora_path: Optional[str] = None
    pinned: Optional[bool] = None
    # False for adapters whose weights arrived over the wire (lora_path
    # "__distributed__" / "__tensor__"): there is no disk artifact to reload
    # from, so they must never be LRU-evicted nor implicitly reloaded.
    # Trailing field with a default keeps the array_like wire format
    # compatible with refs encoded before this field existed.
    reloadable: bool = True
    # Active weight version. Keep this final for positional wire compatibility.
    version: int = 0

    def __post_init__(self):
        if self.lora_id is None:
            raise ValueError("lora_id cannot be None")

    @staticmethod
    def deterministic_id(lora_name: str, lora_path: str) -> str:
        """Stable ``lora_id`` for ``--lora-paths`` adapters.

        Each node in a multi-node launch parses ``--lora-paths`` independently;
        ``uuid4`` would mint a different id per node for the same adapter,
        breaking cross-node lookups when the master broadcasts a request id.
        """
        return uuid5(NAMESPACE_URL, f"{lora_name}\0{lora_path}").hex

    def __str__(self) -> str:
        parts = [
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class LoRARegistry:
    """
    The central registry to keep track of available LoRA adapters and ongoing LoRA requests.

    The `LoRARegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available LoRA adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    """

    def __init__(self, lora_paths: Optional[List[LoRARef]] = None):
        assert lora_paths is None or all(
            isinstance(lora, LoRARef) for lora in lora_paths
        ), (
            "server_args.lora_paths should have been normalized to LoRARef objects during server initialization. "
            "Please file an issue if you see this error."
        )

        # Admission snapshots and lease increments use the writer lock so an
        # unload cannot race between lookup and counter acquisition.
        self._registry_lock = RWLock()
        # An ordered dictionary to hold LoRARef objects, mapping from LoRA name to LoRARef.
        # The LoRARefs are stored in LRU order, such that LoRA adapters that have been
        # most recently used are stored at the end. Note that lookups count for accesses.
        # Ties are broken arbitrarily.
        self._registry: OrderedDict[str, LoRARef] = OrderedDict()
        # Counters for ongoing requests, mapping from LoRA ID to ConcurrentCounter.
        self._counters: Dict[str, ConcurrentCounter] = {}

        # Initialize the registry with provided LoRA paths, if present.
        if lora_paths:
            for lora_ref in lora_paths:
                self._register_adapter(lora_ref)

    async def register(self, lora_ref: LoRARef):
        """
        Register a new LoRARef object in the registry.

        Args:
            lora_ref (LoRARef): The LoRARef object to register.
        """
        async with self._registry_lock.writer_lock:
            self._register_adapter(lora_ref)

    async def unregister(self, lora_name: str) -> str:
        """
        Unregister a LoRARef object from the registry and returns the removed LoRA ID.

        Args:
            lora_name (str): The name of the LoRA model to unregister.
        """
        async with self._registry_lock.writer_lock:
            lora_ref = self._registry.get(lora_name, None)
            if lora_ref is None:
                raise ValueError(
                    f"LoRA with name {lora_name} does not exist. Loaded LoRAs: {self._registry.keys()}"
                )
            del self._registry[lora_name]

        return lora_ref.lora_id

    async def get_lora_id(self, lora_name: str) -> Optional[str]:
        """Return the ``lora_id`` of a registered adapter, or ``None``."""
        async with self._registry_lock.reader_lock:
            lora_ref = self._registry.get(lora_name, None)
            return lora_ref.lora_id if lora_ref is not None else None

    async def register_or_reuse(
        self,
        lora_ref: LoRARef,
        upsert: bool = False,
        *,
        preserve_pinned: bool = False,
        bump_version: bool = False,
    ) -> Tuple[LoRARef, bool]:
        """Resolve which identity a load request should use.

        Returns ``(ref, reused)``. With ``upsert`` and a same-name adapter
        already registered, the returned ref adopts the existing ``lora_id``
        (``reused=True``) so the backend refreshes that adapter in place;
        otherwise ``lora_ref`` is returned unchanged (``reused=False``).
        Nothing is registered here: the caller commits the resolved ref with
        ``register`` / ``refresh`` once the backend load succeeded, keeping
        failed loads invisible to the registry.
        """
        if not upsert:
            return lora_ref, False
        async with self._registry_lock.reader_lock:
            existing = self._registry.get(lora_ref.lora_name, None)
            if existing is None:
                return lora_ref, False
            updates: Dict[str, object] = {"lora_id": existing.lora_id}
            if preserve_pinned:
                updates["pinned"] = existing.pinned
            if bump_version:
                updates["version"] = existing.version + 1
            return replace(lora_ref, **updates), True

    async def refresh(self, lora_ref: LoRARef):
        """Replace a registered adapter's ref after a successful upsert.

        Keeps the id (asserted) while adopting the new path/pinned metadata,
        and counts as a use for LRU ordering.
        """
        async with self._registry_lock.writer_lock:
            existing = self._registry.get(lora_ref.lora_name, None)
            assert existing is not None and existing.lora_id == lora_ref.lora_id, (
                f"refresh() must target a registered adapter with the same lora_id; "
                f"got {lora_ref}, registered: {existing}"
            )
            self._registry[lora_ref.lora_name] = lora_ref
            self._registry.move_to_end(lora_ref.lora_name)

    def _lookup_refs_for_admission(
        self, lora_name: Union[str, List[Optional[str]]]
    ) -> List[Optional[LoRARef]]:
        if isinstance(lora_name, str):
            names = [lora_name]
        elif isinstance(lora_name, list):
            names = lora_name
        else:
            raise TypeError("lora_name must be either a string or a list of strings.")

        refs = []
        for name in names:
            if name is None:
                refs.append(None)
                continue
            ref = self._registry.get(name)
            if ref is None:
                raise ValueError(
                    f"The following requested LoRA adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(name)
            refs.append(ref)
        return refs

    async def _increment_ref_counters(
        self, refs: List[Optional[LoRARef]]
    ) -> None:
        await asyncio.gather(
            *[
                self._counters[ref.lora_id].increment(notify_all=False)
                for ref in refs
                if ref is not None
            ]
        )

    async def _acquire_refs(
        self, lora_name: Union[str, List[Optional[str]]]
    ) -> List[Optional[LoRARef]]:
        async with self._registry_lock.writer_lock:
            refs = self._lookup_refs_for_admission(lora_name)
            await self._increment_ref_counters(refs)
            return refs

    async def acquire(
        self, lora_name: Union[str, List[Optional[str]]]
    ) -> Union[str, List[Optional[str]]]:
        """Acquire request leases and return the existing ID-only shape."""
        refs = await self._acquire_refs(lora_name)
        ids = [ref.lora_id if ref is not None else None for ref in refs]
        return ids[0] if isinstance(lora_name, str) else ids

    async def acquire_with_version(
        self, lora_name: Union[str, List[Optional[str]]]
    ) -> Union[
        Tuple[str, int],
        Tuple[List[Optional[str]], List[Optional[int]]],
    ]:
        """Acquire leases and atomically snapshot adapter IDs and versions."""
        refs = await self._acquire_refs(lora_name)
        ids = [ref.lora_id if ref is not None else None for ref in refs]
        versions = [ref.version if ref is not None else None for ref in refs]
        if isinstance(lora_name, str):
            return ids[0], versions[0]
        return ids, versions

    async def release(self, lora_id: Union[str, List[str]]):
        """
        Decrements the usage counter for a LoRA adapter, indicating that it is no longer in use.
        """

        async with self._registry_lock.reader_lock:
            if isinstance(lora_id, str):
                await self._counters[lora_id].decrement()
            elif isinstance(lora_id, list):
                await asyncio.gather(
                    *[
                        self._counters[id].decrement()
                        for id in lora_id
                        if id is not None
                    ]
                )
            else:
                raise TypeError("lora_id must be either a string or a list of strings.")

    async def wait_for_unload(self, lora_id: str):
        """
        Waits until the usage counter for a LoRA adapter reaches zero, indicating that it is no longer in use.
        This is useful for ensuring that a LoRA adapter can be safely unloaded.

        This method itself is not synchronized, which is safe because it should only be called during LoRA unloading,
        which itself is guaranteed to be sequential.
        """
        assert (
            lora_id not in self._registry
        ), "wait_for_unload should only be called after the LoRA adapter has been unregistered. "
        assert (
            lora_id in self._counters
        ), "The LoRA ID should still have a counter if it has been registered before."

        # Wait until no requests are using this LoRA adapter.
        await self._counters[lora_id].wait_for_zero()
        del self._counters[lora_id]

    async def get_unregistered_loras(self, lora_name: set[str]):
        """
        Returns all LoRA adapters in lora_name that are not found in self._registry.
        """
        async with self._registry_lock.writer_lock:
            unregistered_loras = []

            for name in lora_name:
                if name in self._registry:
                    # This counts as a lookup, so we want to update the cache
                    self._registry.move_to_end(name)
                else:
                    unregistered_loras.append(name)

            return unregistered_loras

    async def lru_lora_name(self, exclude_pinned=False):
        """
        Returns the least recently used LoRA adapter.
        If exclude_pinned is True, then return the LRU LoRA adapter that isn't pinned.
        """
        async with self._registry_lock.reader_lock:
            for lora_name, lora_ref in self._registry.items():
                if exclude_pinned and lora_ref.pinned:
                    continue
                return lora_name
            return None

    def _register_adapter(self, lora_ref: LoRARef):
        """
        Internal helper method to register a LoRA adapter.
        """

        if lora_ref.lora_name in self._registry:
            raise ValueError(
                f"LoRA with name {lora_ref.lora_name} already exists. Loaded LoRAs: {self._registry.keys()}"
            )
        self._registry[lora_ref.lora_name] = lora_ref
        self._counters[lora_ref.lora_id] = ConcurrentCounter()
        return lora_ref

    @property
    def num_registered_loras(self) -> int:
        """
        Returns the total number of LoRA adapters currently registered.
        """
        return len(self._registry)

    def get_all_adapters(self) -> Dict[str, LoRARef]:
        """
        Returns a dictionary of all registered LoRA adapters.

        Returns:
            Dict[str, LoRARef]: A dictionary mapping LoRA names to LoRARef objects.
        """
        return dict(self._registry)
