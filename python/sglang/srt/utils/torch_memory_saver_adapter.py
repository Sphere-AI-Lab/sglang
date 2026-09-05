import logging
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


def _force_preload_hook_mode():
    """SGLang's pauseable CUDA graph (used for RL weight-sync while the graph is
    captured) requires torch_memory_saver hook_mode="preload" (its default). In
    a COLOCATE setup, importing Megatron flips the SHARED torch_memory_saver
    singleton to hook_mode="torch" at module-import time (Megatron
    training.py / inference dynamic_context.py, unconditionally on import), which
    would make sglang's `_memory_saver.cuda_graph(...)` assert
    "Only hook_mode=preload supports pauseable CUDA Graph currently". Force it
    back to "preload" before the singleton is lazily initialized (on the first
    region/cuda_graph use). No-op once initialized (the setter raises
    "Cannot configure after initialization", which we swallow). This only touches
    sglang's own process; Megatron's training process keeps its own "torch"."""
    if import_error is not None:
        return
    try:
        _memory_saver.hook_mode = "preload"
    except Exception:
        pass


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        if enable:
            _force_preload_hook_mode()
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str, enable_cpu_backup: bool = False):
        # Ensure preload before the singleton is initialized (region is one of
        # the two lazy-init triggers). Guards against Megatron being imported
        # after create() set it (import-order independent).
        _force_preload_hook_mode()
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, **kwargs):
        _force_preload_hook_mode()
        return _memory_saver.cuda_graph(**kwargs)

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
