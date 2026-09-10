"""Fork-owned adapter weight-sync extension, shared by every PEFT method.

Upstream sglang has no counterpart: its adapters are immutable, loaded from disk
and never modified. Reinforcement learning inverts that -- the trainer republishes
one adapter's weights every step while the sampler keeps serving -- so the fork
needs staged transport, versioned activation, and a CUDA-graph-safe in-place slot
refresh. This package owns that machinery once, rather than each method growing
its own copy.

Layout:
    manager.py           method-agnostic lifecycle; per-method work goes through hooks
    mem_pool.py          slot-paged buffers + the stage/activate state machine
    utils.py             helpers the two above need
    tokenizer_backend.py AdapterStagingBackend ABC + the get_staging_backend() registry
                         tokenizer_control_mixin.py dispatches through

Deliberately NOT here: the adapter registry. Registry state (ids, refcounts,
pinning) is adapter IDENTITY, needed for ordinary multi-tenant serving with no
hot-swap at all, and upstream keeps its own registry inside the serving package
(srt/lora/lora_registry.py). Putting it here would make plain serving depend on
this extension. It stays in the serving packages.

Status: WS2-1 copied manager.py and mem_pool.py out of ``srt/oft/base/``.
``srt/oft`` still runs on its own copy of manager.py/mem_pool.py, so those two
remain inert scaffolding until WS2-4 migrates it here. tokenizer_backend.py is
live: TokenizerControlMixin's update_adapter_from_distributed/
activate_adapter_version dispatch through get_staging_backend(), and
srt/lora/staged_manager.py's LoRAStagingBackend is its first implementation.
"""
