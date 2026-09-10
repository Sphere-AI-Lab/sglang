"""Small, OFT-owned adapter-slot eviction policies."""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Set


class EvictionPolicy(ABC):
    @abstractmethod
    def mark_used(self, uid: Optional[str]) -> None:
        pass

    @abstractmethod
    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        pass

    @abstractmethod
    def remove(self, uid: Optional[str]) -> None:
        pass


class LRUEvictionPolicy(EvictionPolicy):
    def __init__(self):
        self.access_order = OrderedDict()

    def mark_used(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.access_order.pop(uid, None)
            self.access_order[uid] = time.monotonic()

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        for uid in self.access_order:
            if uid in candidates:
                return uid
        if None in candidates:
            return None
        raise RuntimeError(f"Failed to select LRU victim from {candidates}")

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.access_order.pop(uid, None)


class FIFOEvictionPolicy(EvictionPolicy):
    def __init__(self):
        self.insertion_order = OrderedDict()

    def mark_used(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.insertion_order.setdefault(uid, None)

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        for uid in self.insertion_order:
            if uid in candidates:
                return uid
        if None in candidates:
            return None
        raise RuntimeError(f"Failed to select FIFO victim from {candidates}")

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.insertion_order.pop(uid, None)


def get_eviction_policy(policy_name: str) -> EvictionPolicy:
    policies = {"fifo": FIFOEvictionPolicy, "lru": LRUEvictionPolicy}
    try:
        return policies[policy_name]()
    except KeyError as error:
        raise ValueError(f"Unknown eviction policy: {policy_name}") from error
