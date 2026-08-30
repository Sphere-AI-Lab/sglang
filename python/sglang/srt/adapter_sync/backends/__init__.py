"""Per-method backends for the shared weight-sync core.

Each backend teaches ``VersionedStaging`` how to move weights for one PEFT
method, without that method's serving package having to change. OFT's staging
still lives in ``srt/oft`` and migrates here in WS2-4.
"""
