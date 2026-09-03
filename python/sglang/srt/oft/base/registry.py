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
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

from sglang.srt.utils import ConcurrentCounter
from sglang.srt.utils.aio_rwlock import RWLock


@dataclass(frozen=True)
class AdapterRef:
    """Generic adapter-reference record. Originally intended to be shared with
    LoRA (see the historical note in git blame), but LoRARef evolved
    independently as its own msgspec.Struct — today only OFTRef subclasses
    this. Holds the unified adapter identity; AdapterRegistry accesses
    adapters only through these members.

    The unique ``oft_id`` eliminates conflicts from reused names/paths and can
    be used to generate deterministic cache keys (e.g. radix cache)."""

    oft_id: str = field(default_factory=lambda: uuid4().hex)
    oft_name: Optional[str] = None
    oft_path: Optional[str] = None
    pinned: Optional[bool] = None
    version: int = 1
    # False for adapters whose weights arrived over the wire (no on-disk
    # artifact to reload from): they must never be LRU-evicted nor
    # implicitly reloaded. Mirrors LoRARef.reloadable exactly.
    reloadable: bool = True

    def __post_init__(self):
        if self.oft_id is None:
            raise ValueError("oft_id cannot be None")
        if self.version < 0:
            raise ValueError("version must be non-negative")


class AdapterRegistry:
    """The central registry to keep track of available adapters and ongoing adapter requests.

    The `AdapterRegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    OFT's registry base (see ``OFTRegistry`` for subclass-specific behavior).
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

        return ref.oft_id

    async def resolve_or_reuse(
        self,
        ref: AdapterRef,
        upsert: bool = False,
        *,
        preserve_pinned: bool = False,
    ) -> Tuple[AdapterRef, bool]:
        """Resolve which identity a load request should use.

        Returns ``(ref, reused)``. With ``upsert`` and a same-name adapter
        already registered, the returned ref adopts the existing
        ``oft_id`` (``reused=True``) so the backend refreshes that
        adapter in place; otherwise ``ref`` is returned unchanged
        (``reused=False``). Nothing is registered here: the caller commits
        the resolved ref with ``register``/``refresh`` once the backend load
        succeeded, keeping failed loads invisible to the registry.

        Also bumps ``version`` past the existing entry's on reuse:
        the radix cache key is extended with the adapter's version (see
        ``maybe_extend_extra_key``), so KV produced under the old weights
        must live under a different key than requests arriving after this
        in-place refresh -- otherwise a prompt re-served after an upsert
        could silently return output computed with the stale, pre-upsert
        weights via a cached KV prefix.
        """
        if not upsert:
            return ref, False
        async with self._registry_lock.reader_lock:
            existing = self._registry.get(ref.oft_name)
            if existing is None:
                return ref, False
            updates = {
                "oft_id": existing.oft_id,
                "version": existing.version + 1,
            }
            if preserve_pinned:
                updates["pinned"] = existing.pinned
            return replace(ref, **updates), True

    async def refresh(self, ref: AdapterRef):
        """Replace a registered adapter's ref after a successful upsert.

        Keeps the id (asserted) while adopting the new path/pinned metadata,
        and counts as a use for LRU ordering.
        """
        async with self._registry_lock.writer_lock:
            existing = self._registry.get(ref.oft_name)
            assert existing is not None and existing.oft_id == ref.oft_id, (
                f"refresh() must target a registered adapter with the same "
                f"oft_id; got {ref}, registered: {existing}"
            )
            self._registry[ref.oft_name] = ref
            self._registry.move_to_end(ref.oft_name)

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
            return ref.oft_id

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

    def _lookup_refs_for_admission(
        self, name: Union[str, List[Optional[str]]]
    ) -> List[Optional[AdapterRef]]:
        """Lookup refs, handling both str and list inputs, normalizing to list output."""
        if isinstance(name, str):
            names = [name]
        elif isinstance(name, list):
            names = name
        else:
            raise TypeError("name must be either a string or a list of strings.")

        refs = []
        for n in names:
            if n is None:
                refs.append(None)
                continue
            ref = self._registry.get(n)
            if ref is None:
                raise ValueError(
                    f"The following requested adapters are not loaded: {n}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            self._registry.move_to_end(n)
            refs.append(ref)
        return refs

    async def _increment_ref_counters(
        self, refs: List[Optional[AdapterRef]]
    ) -> None:
        """Increment usage counters for non-None refs."""
        await asyncio.gather(
            *[
                self._counters[ref.oft_id].increment(notify_all=False)
                for ref in refs
                if ref is not None
            ]
        )

    async def _acquire_refs(
        self, name: Union[str, List[Optional[str]]]
    ) -> List[Optional[AdapterRef]]:
        """Atomically snapshot the matching AdapterRef(s) and start tracking usage.

        Lookup, version snapshot, and counter increment are one atomic admission
        step. An unload cannot remove the adapter between them.
        """
        # Lookup, version snapshot, and counter increment are one atomic
        # admission step. An unload cannot remove the adapter between them.
        async with self._registry_lock.writer_lock:
            refs = self._lookup_refs_for_admission(name)
            await self._increment_ref_counters(refs)
            return refs

    async def acquire_with_version(
        self, name: Union[str, List[Optional[str]]]
    ) -> Union[
        Tuple[Optional[str], Optional[int]],
        Tuple[List[Optional[str]], List[Optional[int]]],
    ]:
        """Acquire request leases and atomically snapshot ids and versions."""
        refs = await self._acquire_refs(name)
        ids = [ref.oft_id if ref is not None else None for ref in refs]
        versions = [ref.version if ref is not None else None for ref in refs]
        if isinstance(name, str):
            return ids[0], versions[0]
        return ids, versions

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

        if ref.oft_name in self._registry:
            raise ValueError(
                f"adapter with name {ref.oft_name} already exists. Loaded adapters: {self._registry.keys()}"
            )
        self._registry[ref.oft_name] = ref
        self._counters[ref.oft_id] = ConcurrentCounter()
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
