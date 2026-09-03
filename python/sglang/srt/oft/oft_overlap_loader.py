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
        self, adapter_id: Optional[str], running_ofts: set[Optional[str]]
    ) -> bool:
        """Start or poll asynchronous loading for one OFT adapter."""
        self._drain_completed_overlap_loads()

        load_status = self._check_overlap_load_status(adapter_id)
        if load_status == OFTOverlapLoadStatus.LOADING:
            return False
        if load_status == OFTOverlapLoadStatus.NOT_LOADED:
            if self._try_start_overlap_load(adapter_id, running_ofts):
                logger.debug("Loading OFT adapter %s asynchronously", adapter_id)
            return False

        assert load_status == OFTOverlapLoadStatus.LOADED
        return True

    def _check_overlap_load_status(
        self, adapter_id: Optional[str]
    ) -> OFTOverlapLoadStatus:
        if adapter_id in self.oft_to_overlap_load_event:
            return OFTOverlapLoadStatus.LOADING
        if adapter_id in self.oft_manager.memory_pool.uid_to_buffer_id:
            return OFTOverlapLoadStatus.LOADED
        return OFTOverlapLoadStatus.NOT_LOADED

    def _drain_completed_overlap_loads(self) -> None:
        completed_loads = [
            (adapter_id, event)
            for adapter_id, event in self.oft_to_overlap_load_event.items()
            if event.query()
        ]
        for adapter_id, event in completed_loads:
            torch.cuda.current_stream().wait_event(event)
            del self.oft_to_overlap_load_event[adapter_id]

    def _try_start_overlap_load(
        self, adapter_id: Optional[str], running_ofts: set[Optional[str]]
    ) -> bool:
        ofts_to_be_loaded = running_ofts | self.oft_to_overlap_load_event.keys()
        new_oft_set = {adapter_id} | ofts_to_be_loaded
        if not self.oft_manager.validate_oft_batch(new_oft_set):
            return False

        with self.load_stream_context:
            self.oft_manager.fetch_new_ofts({adapter_id}, ofts_to_be_loaded)
            event = self.device_module.Event()
            event.record(self.load_stream)

        self.oft_to_overlap_load_event[adapter_id] = event
        return True
