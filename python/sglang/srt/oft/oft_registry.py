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
from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Union
from uuid import uuid4

from sglang.srt.utils import ConcurrentCounter
from sglang.srt.utils.aio_rwlock import RWLock


@dataclass(frozen=True)
class OFTRef:
    """
    Reference record for an OFT model.

    This object guarantees a unique ``oft_id`` and may include ``oft_name``, ``oft_path``, and ``pinned``.
    The ID eliminates conflicts from reused OFT names or paths and can be used to generate deterministic cache
    keys (e.g., radix cache).
    """

    oft_id: str = field(default_factory=lambda: uuid4().hex)
    oft_name: Optional[str] = None
    oft_path: Optional[str] = None
    pinned: Optional[bool] = None
    oft_version: int = 1

    def __post_init__(self):
        if self.oft_id is None:
            raise ValueError("oft_id cannot be None")
        if self.oft_version < 0:
            raise ValueError("oft_version must be non-negative")

    def __str__(self) -> str:
        parts = [
            f"{f.name}={value}"
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


class OFTRegistry:
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
            "server_args.oft_paths should have been normalized to OFTRef objects during server initialization. "
            "Please file an issue if you see this error."
        )

        # A read-write lock to ensure adapters loading / unloading operations are exclusive.
        # Please note that the counter increment/decrement operations are not synchronized through this
        # lock, as they are designed to be non-blocking and can be performed concurrently.
        self._registry_lock = RWLock()
        # An ordered dictionary to hold OFTRef objects, mapping from OFT name to OFTRef.
        # The OFTRefs are stored in LRU order, such that OFT adapters that have been
        # most recently used are stored at the end. Note that lookups count for accesses.
        # Ties are broken arbitrarily.
        self._registry: OrderedDict[str, OFTRef] = OrderedDict()
        # Counters for ongoing requests, mapping from OFT ID to ConcurrentCounter.
        self._counters: Dict[str, ConcurrentCounter] = {}

        # Initialize the registry with provided OFT paths, if present.
        if oft_paths:
            for oft_ref in oft_paths:
                self._register_adapter(oft_ref)

    async def register(self, oft_ref: OFTRef):
        """
        Register a new OFTRef object in the registry.

        Args:
            oft_ref (OFTRef): The OFTRef object to register.
        """
        async with self._registry_lock.writer_lock:
            self._register_adapter(oft_ref)

    async def unregister(self, oft_name: str) -> str:
        """
        Unregister an OFTRef object from the registry and returns the removed OFT ID.

        Args:
            oft_name (str): The name of the OFT model to unregister.
        """
        async with self._registry_lock.writer_lock:
            oft_ref = self._registry.get(oft_name, None)
            if oft_ref is None:
                raise ValueError(
                    f"OFT with name {oft_name} does not exist. Loaded OFTs: {self._registry.keys()}"
                )
            del self._registry[oft_name]

        return oft_ref.oft_id

    async def replace(self, oft_ref: OFTRef) -> Optional[str]:
        """Atomically route future acquires for oft_ref.oft_name to a new OFTRef.

        Returns the old OFT ID if the public name was already active, or None.
        The old counter is intentionally kept so in-flight requests can release
        their reference. The caller is responsible for invoking
        ``wait_for_unload(old_id)`` once in-flight requests drain, to free the
        counter; otherwise the old id's counter leaks indefinitely.
        """
        async with self._registry_lock.writer_lock:
            old_ref = self._registry.get(oft_ref.oft_name)
            if old_ref is not None:
                del self._registry[oft_ref.oft_name]
            self._register_adapter(oft_ref)
            return old_ref.oft_id if old_ref is not None else None

    async def acquire(self, oft_name: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Queries registry for OFT IDs based on OFT names and start tracking the usage of the corresponding OFT adapters
        by incrementing its counter.
        """

        def _lookup(name: str) -> str:
            if name is None:
                return None

            oft_ref = self._registry.get(name, None)
            if oft_ref is None:
                raise ValueError(
                    f"The following requested OFT adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(name)
            return oft_ref.oft_id

        if isinstance(oft_name, str):
            async with self._registry_lock.writer_lock:
                oft_id = _lookup(oft_name)

            await self._counters[oft_id].increment(notify_all=False)
            return oft_id
        elif isinstance(oft_name, list):
            async with self._registry_lock.writer_lock:
                oft_ids = [_lookup(name) for name in oft_name]

            # Increment the counters only after all IDs are looked up.
            await asyncio.gather(
                *[
                    self._counters[id].increment(notify_all=False)
                    for id in oft_ids
                    if id is not None
                ]
            )
            return oft_ids
        else:
            raise TypeError("oft_name must be either a string or a list of strings.")

    async def get_version_by_id(
        self, oft_id: Union[str, List[str], None]
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
                if oft_ref.oft_id == uid:
                    return oft_ref.oft_version
            raise ValueError(f"OFT ID {uid} does not exist.")

        async with self._registry_lock.reader_lock:
            if isinstance(oft_id, str) or oft_id is None:
                return _lookup(oft_id)
            if isinstance(oft_id, list):
                return [_lookup(uid) for uid in oft_id]
            raise TypeError("oft_id must be None, a string, or a list of strings.")

    async def bump_version_by_id(self, oft_id: str) -> OFTRef:
        """
        Increment the version for an already-registered adapter ID.

        This keeps the stable ``oft_id`` used by the OFT memory pool while
        invalidating radix-cache keys after streamed on-policy updates.
        """

        async with self._registry_lock.writer_lock:
            for oft_name, oft_ref in self._registry.items():
                if oft_ref.oft_id == oft_id:
                    new_ref = replace(oft_ref, oft_version=oft_ref.oft_version + 1)
                    self._registry[oft_name] = new_ref
                    return new_ref
        raise ValueError(f"OFT ID {oft_id} does not exist.")

    async def release(self, oft_id: Union[str, List[str]]):
        """
        Decrements the usage counter for an OFT adapter, indicating that it is no longer in use.
        """

        async with self._registry_lock.reader_lock:
            if isinstance(oft_id, str):
                await self._counters[oft_id].decrement()
            elif isinstance(oft_id, list):
                await asyncio.gather(
                    *[
                        self._counters[id].decrement()
                        for id in oft_id
                        if id is not None
                    ]
                )
            else:
                raise TypeError("oft_id must be either a string or a list of strings.")

    async def wait_for_unload(self, oft_id: str):
        """
        Waits until the usage counter for an OFT adapter reaches zero, indicating that it is no longer in use.
        This is useful for ensuring that an OFT adapter can be safely unloaded.

        This method itself is not synchronized, which is safe because it should only be called during OFT unloading,
        which itself is guaranteed to be sequential.
        """
        assert (
            oft_id not in self._registry
        ), "wait_for_unload should only be called after the OFT adapter has been unregistered. "
        assert (
            oft_id in self._counters
        ), "The OFT ID should still have a counter if it has been registered before."

        # Wait until no requests are using this OFT adapter.
        await self._counters[oft_id].wait_for_zero()
        del self._counters[oft_id]

    async def get_unregistered_ofts(self, oft_name: set[str]):
        """
        Returns all OFT adapters in oft_name that are not found in self._registry.
        """
        async with self._registry_lock.writer_lock:
            unregistered_ofts = []

            for name in oft_name:
                if name in self._registry:
                    # This counts as a lookup, so we want to update the cache
                    self._registry.move_to_end(name)
                else:
                    unregistered_ofts.append(name)

            return unregistered_ofts

    async def lru_oft_name(self, exclude_pinned=False):
        """
        Returns the least recently used OFT adapter.
        If exclude_pinned is True, then return the LRU OFT adapter that isn't pinned.
        """
        async with self._registry_lock.reader_lock:
            if not exclude_pinned:
                return next(iter(self._registry), None)

            for oft_name, oft_ref in self._registry.items():
                if not oft_ref.pinned:
                    return oft_name
            else:
                return None

    def _register_adapter(self, oft_ref: OFTRef):
        """
        Internal helper method to register an OFT adapter.
        """

        if oft_ref.oft_name in self._registry:
            raise ValueError(
                f"OFT with name {oft_ref.oft_name} already exists. Loaded OFTs: {self._registry.keys()}"
            )
        self._registry[oft_ref.oft_name] = oft_ref
        self._counters[oft_ref.oft_id] = ConcurrentCounter()
        return oft_ref

    @property
    def num_registered_ofts(self) -> int:
        """
        Returns the total number of OFT adapters currently registered.
        """
        return len(self._registry)

    def get_all_adapters(self) -> Dict[str, OFTRef]:
        """
        Returns a dictionary of all registered OFT adapters.

        Returns:
            Dict[str, OFTRef]: A dictionary mapping OFT names to OFTRef objects.
        """
        return dict(self._registry)
