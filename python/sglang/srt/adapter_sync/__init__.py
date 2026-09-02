"""Fork-owned adapter weight-sync extension, shared by every PEFT method.

Upstream sglang has no counterpart: its adapters are immutable, loaded from disk
and never modified. Reinforcement learning inverts that -- the trainer republishes
one adapter's weights every step while the sampler keeps serving -- so the fork
needs staged transport and versioned activation.

Layout:
    tokenizer_backend.py AdapterStagingBackend ABC + the get_staging_backend()
                         registry tokenizer_control_mixin.py dispatches
                         through. LoRA (srt/lora/staged_manager.py) and OFT
                         (srt/oft/staged_manager.py) each implement it.

Deliberately NOT here: the adapter registry (identity/refcounts/pinning) or
the memory-pool/lifecycle machinery -- each stays in its own serving package
(srt/lora, srt/oft), which already implement it against their own buffer
layouts.
"""
