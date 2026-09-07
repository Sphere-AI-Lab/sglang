"""Receive one inactive-slot OFT update without occupying the decode scheduler."""

from concurrent.futures import ThreadPoolExecutor

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.managers.io_struct import UpdateAdapterFromDistributedReqOutput


class AsyncAdapterStage:
    def __init__(self, device_index, tp_cpu_group=None):
        self.device_index = device_index
        self.stream = torch.cuda.Stream(device=device_index)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="oft-stage"
        )
        self.future = None
        self.request = None
        self.start_forward_ct = 0
        # Only TP0 sends the HTTP response. A separate CPU group makes that
        # response contingent on ALL ranks finishing, without interleaving
        # worker collectives with the scheduler's control broadcasts.
        self.completion_group = None
        if dist.is_initialized() and dist.get_world_size(tp_cpu_group) > 1:
            self.completion_group = dist.new_group(
                ranks=dist.get_process_group_ranks(tp_cpu_group),
                backend="gloo",
                use_local_synchronization=True,
            )

    def submit(self, request, stage_fn, forward_ct):
        if self.future is not None:
            raise RuntimeError("An OFT stage is already pending")
        # Protect startup/state initialization, but do not insert a dependency
        # from subsequent decode forwards to the staging stream.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device_index))
        self.request = request
        self.start_forward_ct = forward_ct
        self.future = self.executor.submit(self._run, stage_fn, request, ready)

    def _run(self, stage_fn, request, ready):
        try:
            with torch.cuda.device(self.device_index), torch.cuda.stream(
                self.stream
            ), torch.no_grad():
                self.stream.wait_event(ready)
                output = stage_fn(request)
                # This blocks this worker only, never the decode scheduler.
                self.stream.synchronize()
        except Exception as exc:
            output = UpdateAdapterFromDistributedReqOutput(
                success=False,
                message=f"Asynchronous OFT staging failed: {exc}",
                adapter_version=request.adapter_version,
                weight_version=request.weight_version,
            )
        if self.completion_group is not None:
            outcomes = [None] * dist.get_world_size(self.completion_group)
            dist.all_gather_object(
                outcomes, (output.success, output.message), group=self.completion_group
            )
            errors = [message for success, message in outcomes if not success]
            if errors:
                output = msgspec.structs.replace(
                    output,
                    success=False,
                    message="; ".join(errors),
                    staged_adapter_version=None,
                )
        return output

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, request = self.future, self.request
        self.future = self.request = None
        return request, future.result()
