"""Orbit PEFT package: the thin integration seams into upstream SGLang for OFT
(Orthogonal Finetuning) adapters. The OFT serving implementations themselves
live in the sibling package ``sglang.srt.oft`` (see ``--oft-impl``: 'sibling'
or 'staged'); this package owns the façade ``model_runner.py`` calls through
(``peft/integration.py``). The CLI-flag surface lives in ``oft/config.py``.
"""
