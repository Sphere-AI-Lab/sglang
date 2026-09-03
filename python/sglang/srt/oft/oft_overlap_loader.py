"""OFT counterpart of ``lora/lora_overlap_loader.py`` -- ported field-for-
field (``lora_id`` -> ``oft_id``, ``LoRAManager`` -> ``OFTManager``,
``pending_lora_load_events`` -> ``pending_oft_load_events``). This is NOT a
disk-loading mechanism: it overlaps the GPU-side cost of materializing a
not-yet-resident adapter's weights into a buffer-pool slot with ongoing
compute, on a separate CUDA stream, so admission doesn't synchronously stall
the forward pass that first names the adapter -- the same latency
``ForwardBatch.init_new``'s unconditional ``fetch_new_ofts`` call otherwise
pays inline every time.
"""

import logging
from enum import Enum, auto
from typing import Dict, Optional

import torch
from torch.cuda import Event as CudaEvent
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext

from sglang.srt.oft.oft_manager import OFTManager

logger = logging.getLogger(__name__)


class OFTOverlapLoadStatus(Enum):
    LOADED = auto()
    LOADING = auto()
    NOT_LOADED = auto()


class OFTOverlapLoader:
    def __init__(self, oft_manager):
        self.oft_manager: OFTManager = oft_manager
        self.device_module = torch.get_device_module(self.oft_manager.device)
        self.load_stream: CudaStream = self.device_module.Stream()
        self.load_stream_context: CudaStreamContext = self.device_module.stream(
            self.load_stream
        )
        self.oft_to_overlap_load_event: Dict[Optional[str], CudaEvent] = (
            self.oft_manager.pending_oft_load_events
        )

    def try_overlap_load_oft(
        self, oft_id: Optional[str], running_ofts: set[Optional[str]]
    ) -> bool:
        """
        Check an OFT adapter's asynchronous load status, and try to load it if there's capacity
        in the memory pool. Returns whether or not the adapter has been loaded.
        """
        # Drain completed async loads before status/capacity checks so finished
        # adapters no longer count as in-flight.
        self._drain_completed_overlap_loads()

        oft_load_status = self._check_overlap_load_status(oft_id)
        if oft_load_status == OFTOverlapLoadStatus.LOADING:
            return False
        elif oft_load_status == OFTOverlapLoadStatus.NOT_LOADED:
            res = self._try_start_overlap_load(oft_id, running_ofts)
            if res:
                logger.debug(f"Loading OFT adapter {oft_id} asynchronously")

            return False
        else:
            assert oft_load_status == OFTOverlapLoadStatus.LOADED
            return True

    def _check_overlap_load_status(
        self, oft_id: Optional[str]
    ) -> OFTOverlapLoadStatus:
        if oft_id in self.oft_to_overlap_load_event:
            return OFTOverlapLoadStatus.LOADING

        # After completed events have been drained, a memory-pool entry with no
        # pending event is safe to use on the current stream.
        if oft_id in self.oft_manager.memory_pool.uid_to_buffer_id:
            return OFTOverlapLoadStatus.LOADED

        return OFTOverlapLoadStatus.NOT_LOADED

    def _drain_completed_overlap_loads(self) -> None:
        completed_loads = [
            (oft_id, event)
            for oft_id, event in self.oft_to_overlap_load_event.items()
            if event.query()
        ]
        for oft_id, event in completed_loads:
            torch.cuda.current_stream().wait_event(event)
            del self.oft_to_overlap_load_event[oft_id]

    def _try_start_overlap_load(
        self, oft_id: Optional[str], running_ofts: set[Optional[str]]
    ) -> bool:
        ofts_to_be_loaded = running_ofts | self.oft_to_overlap_load_event.keys()

        new_oft_set = {oft_id} | ofts_to_be_loaded
        if not self.oft_manager.validate_oft_batch(new_oft_set):
            return False

        with self.load_stream_context:
            self.oft_manager.fetch_new_ofts({oft_id}, ofts_to_be_loaded)
            event = self.device_module.Event()
            event.record(self.load_stream)

        self.oft_to_overlap_load_event[oft_id] = event
        return True
