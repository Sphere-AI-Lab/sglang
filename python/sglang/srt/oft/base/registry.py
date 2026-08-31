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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from uuid import uuid4

from sglang.srt.utils import ConcurrentCounter
from sglang.srt.utils.aio_rwlock import RWLock


@dataclass(frozen=True)
class AdapterRef:
    """Generic adapter-reference record. Originally intended to be shared with
    LoRA, but LoRARef evolved independently as its own ``msgspec.Struct`` --
    today only OFTRef subclasses this. Holds the unified adapter identity;
    AdapterRegistry accesses adapters only through these members.

    The unique ``adapter_id`` eliminates conflicts from reused names/paths and can
    be used to generate deterministic cache keys (e.g. radix cache)."""

    adapter_id: str = field(default_factory=lambda: uuid4().hex)
    adapter_name: Optional[str] = None
    adapter_path: Optional[str] = None
    pinned: Optional[bool] = None
    adapter_version: int = 1

    def __post_init__(self):
        if self.adapter_id is None:
            raise ValueError("adapter_id cannot be None")
        if self.adapter_version < 0:
            raise ValueError("adapter_version must be non-negative")


class AdapterRegistry:
    """
    The central registry to keep track of available adapters and ongoing adapter requests.

    The `AdapterRegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    """

    def __init__(self, adapters: Optional[List[AdapterRef]] = None):
        # A read-write lock to ensure adapters loading / unloading operations are exclusive.
        # Please note that the counter increment/decrement operations are not synchronized through this
        # lock, as they are designed to be non-blocking and can be performed concurrently.
        self._registry_lock = RWLock()
        # An ordered dictionary to hold AdapterRef objects, mapping from adapter name to AdapterRef.
        # The AdapterRefs are stored in LRU order, such that adapters that have been
        # most recently used are stored at the end. Note that lookups count for accesses.
        # Ties are broken arbitrarily.
        self._registry: OrderedDict[str, AdapterRef] = OrderedDict()
        # Counters for ongoing requests, mapping from adapter ID to ConcurrentCounter.
        self._counters: Dict[str, ConcurrentCounter] = {}

        # Initialize the registry with the provided adapters, if present.
        if adapters:
            for ref in adapters:
                self._register_adapter(ref)

    async def register(self, ref: AdapterRef):
        """
        Register a new AdapterRef object in the registry.

        Args:
            ref (AdapterRef): The AdapterRef object to register.
        """
        async with self._registry_lock.writer_lock:
            self._register_adapter(ref)

    async def unregister(self, name: str) -> str:
        """
        Unregister an AdapterRef object from the registry and returns the removed adapter ID.

        Args:
            name (str): The name of the adapter to unregister.
        """
        async with self._registry_lock.writer_lock:
            ref = self._registry.get(name, None)
            if ref is None:
                raise ValueError(
                    f"adapter with name {name} does not exist. Loaded adapters: {self._registry.keys()}"
                )
            del self._registry[name]

        return ref.adapter_id

    async def replace(self, ref: AdapterRef) -> Optional[str]:
        """Atomically route future acquires for ref.adapter_name to a new AdapterRef.

        Returns the old adapter ID if the public name was already active, or None.
        The old counter is intentionally kept so in-flight requests can release
        their reference. The caller is responsible for invoking
        ``wait_for_unload(old_id)`` once in-flight requests drain, to free the
        counter; otherwise the old id's counter leaks indefinitely.
        """
        async with self._registry_lock.writer_lock:
            old_ref = self._registry.get(ref.adapter_name)
            if old_ref is not None:
                del self._registry[ref.adapter_name]
            self._register_adapter(ref)
            return old_ref.adapter_id if old_ref is not None else None

    async def acquire(self, name: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Queries registry for adapter IDs based on adapter names and start tracking the usage of the corresponding
        adapters by incrementing its counter.
        """

        def _lookup(n: str) -> str:
            if n is None:
                return None

            ref = self._registry.get(n, None)
            if ref is None:
                raise ValueError(
                    f"The following requested adapters are not loaded: {n}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(n)
            return ref.adapter_id

        if isinstance(name, str):
            async with self._registry_lock.writer_lock:
                uid = _lookup(name)

            await self._counters[uid].increment(notify_all=False)
            return uid
        elif isinstance(name, list):
            async with self._registry_lock.writer_lock:
                uids = [_lookup(n) for n in name]

            # Increment the counters only after all IDs are looked up.
            await asyncio.gather(
                *[
                    self._counters[id].increment(notify_all=False)
                    for id in uids
                    if id is not None
                ]
            )
            return uids
        else:
            raise TypeError("name must be either a string or a list of strings.")

    async def release(self, uid: Union[str, List[str]]):
        """
        Decrements the usage counter for an adapter, indicating that it is no longer in use.
        """

        async with self._registry_lock.reader_lock:
            if isinstance(uid, str):
                await self._counters[uid].decrement()
            elif isinstance(uid, list):
                await asyncio.gather(
                    *[
                        self._counters[id].decrement()
                        for id in uid
                        if id is not None
                    ]
                )
            else:
                raise TypeError("uid must be either a string or a list of strings.")

    async def wait_for_unload(self, uid: str):
        """
        Waits until the usage counter for an adapter reaches zero, indicating that it is no longer in use.
        This is useful for ensuring that an adapter can be safely unloaded.

        This method itself is not synchronized, which is safe because it should only be called during adapter
        unloading, which itself is guaranteed to be sequential.
        """
        assert (
            uid not in self._registry
        ), "wait_for_unload should only be called after the adapter has been unregistered. "
        assert (
            uid in self._counters
        ), "The adapter ID should still have a counter if it has been registered before."

        # Wait until no requests are using this adapter.
        await self._counters[uid].wait_for_zero()
        del self._counters[uid]

    async def get_unregistered_adapters(self, name: set[str]):
        """
        Returns all adapters in name that are not found in self._registry.
        """
        async with self._registry_lock.writer_lock:
            unregistered_adapters = []

            for n in name:
                if n in self._registry:
                    # This counts as a lookup, so we want to update the cache
                    self._registry.move_to_end(n)
                else:
                    unregistered_adapters.append(n)

            return unregistered_adapters

    async def lru_adapter_name(self, exclude_pinned=False):
        """
        Returns the least recently used adapter.
        If exclude_pinned is True, then return the LRU adapter that isn't pinned.
        """
        async with self._registry_lock.reader_lock:
            if not exclude_pinned:
                return next(iter(self._registry), None)

            for name, ref in self._registry.items():
                if not ref.pinned:
                    return name
            else:
                return None

    def _register_adapter(self, ref: AdapterRef):
        """
        Internal helper method to register an adapter.
        """

        if ref.adapter_name in self._registry:
            raise ValueError(
                f"adapter with name {ref.adapter_name} already exists. Loaded adapters: {self._registry.keys()}"
            )
        self._registry[ref.adapter_name] = ref
        self._counters[ref.adapter_id] = ConcurrentCounter()
        return ref

    @property
    def num_registered_adapters(self) -> int:
        """
        Returns the total number of adapters currently registered.
        """
        return len(self._registry)

    def get_all_adapters(self) -> Dict[str, AdapterRef]:
        """
        Returns a dictionary of all registered adapters.

        Returns:
            Dict[str, AdapterRef]: A dictionary mapping adapter names to AdapterRef objects.
        """
        return dict(self._registry)
