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
        self.oft_to_overlap_load_event: Dict[Optional[str], CudaEvent] = {}

    def try_overlap_load_oft(
        self, oft_id: Optional[str], running_ofts: set[Optional[str]]
    ) -> bool:
        """
        Check an OFT adapter's asynchronous load status, and try to load it if there's capacity
        in the memory pool. Returns whether or not the adapter has been loaded.
        """
        oft_pipeline_load_status = self._check_overlap_load_status(oft_id)
        if oft_pipeline_load_status == OFTOverlapLoadStatus.LOADING:
            return False
        elif oft_pipeline_load_status == OFTOverlapLoadStatus.NOT_LOADED:
            res = self._try_start_overlap_load(oft_id, running_ofts)
            if res:
                logger.debug(f"Loading OFT adapter {oft_id} asynchronously")

            return False
        else:
            assert oft_pipeline_load_status == OFTOverlapLoadStatus.LOADED
            return True

    def _check_overlap_load_status(
        self, oft_id: Optional[str]
    ) -> OFTOverlapLoadStatus:
        if oft_id not in self.oft_to_overlap_load_event:
            return OFTOverlapLoadStatus.NOT_LOADED

        event = self.oft_to_overlap_load_event[oft_id]

        if not event.query():
            return OFTOverlapLoadStatus.LOADING

        torch.cuda.current_stream().wait_event(event)
        del self.oft_to_overlap_load_event[oft_id]

        return OFTOverlapLoadStatus.LOADED

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
