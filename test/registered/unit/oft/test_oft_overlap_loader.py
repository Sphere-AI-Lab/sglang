from types import SimpleNamespace
from unittest import mock


def make_manager():
    manager = SimpleNamespace(
        device="cuda:0",
        memory_pool=SimpleNamespace(uid_to_buffer_id={}),
        pending_oft_load_events={},
    )
    manager.validate_oft_batch = mock.Mock(return_value=True)

    def fetch_new_ofts(new_ofts, _running_ofts):
        for adapter_id in new_ofts:
            manager.memory_pool.uid_to_buffer_id[adapter_id] = len(
                manager.memory_pool.uid_to_buffer_id
            )

    manager.fetch_new_ofts = mock.Mock(side_effect=fetch_new_ofts)
    return manager


def make_loader(manager):
    from sglang.srt.oft.oft_overlap_loader import OFTOverlapLoader

    device_module = mock.MagicMock()
    device_module.Stream.return_value = mock.MagicMock()
    device_module.stream.return_value = mock.MagicMock()
    device_module.Event.return_value = mock.MagicMock()

    with mock.patch(
        "sglang.srt.oft.oft_overlap_loader.torch.get_device_module",
        return_value=device_module,
    ):
        loader = OFTOverlapLoader(manager)
    return loader, device_module


def test_overlap_load_waits_for_completion_before_admission():
    manager = make_manager()
    loader, device_module = make_loader(manager)

    assert not loader.try_overlap_load_oft("adapter-a", set())
    event = loader.oft_to_overlap_load_event["adapter-a"]
    event.query.return_value = False
    assert not loader.try_overlap_load_oft("adapter-a", set())

    event.query.return_value = True
    with mock.patch(
        "sglang.srt.oft.oft_overlap_loader.torch.cuda.current_stream"
    ) as current_stream:
        assert loader.try_overlap_load_oft("adapter-a", set())

    current_stream.return_value.wait_event.assert_called_once_with(event)
    assert "adapter-a" not in loader.oft_to_overlap_load_event
    manager.fetch_new_ofts.assert_called_once_with({"adapter-a"}, set())
    assert device_module.Event.call_count == 1


def test_overlap_validation_includes_running_and_pending_adapters():
    manager = make_manager()
    loader, device_module = make_loader(manager)
    pending_event = mock.MagicMock()
    pending_event.query.return_value = False
    device_module.Event.side_effect = [pending_event, mock.MagicMock()]

    assert loader._try_start_overlap_load("pending", set())
    manager.validate_oft_batch.reset_mock()
    assert not loader.try_overlap_load_oft("new", {"running"})

    manager.validate_oft_batch.assert_called_once_with({"pending", "running", "new"})


def test_oft_admission_delegates_new_adapter_to_overlap_loader():
    from sglang.srt.oft.integration import maybe_admit_request

    overlap_loader = mock.Mock()
    overlap_loader.try_overlap_load_oft.return_value = False
    capacity = mock.Mock()
    scheduler = SimpleNamespace(
        oft_drainer=None,
        enable_oft_overlap_loading=True,
        oft_overlap_loader=overlap_loader,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(oft_manager=capacity)
        ),
    )

    assert not maybe_admit_request(
        scheduler,
        SimpleNamespace(adapter_id="new"),
        {"running"},
    )
    overlap_loader.try_overlap_load_oft.assert_called_once_with("new", {"running"})
    capacity.validate_oft_batch.assert_not_called()


def test_oft_manager_synchronizes_pending_load_before_unload():
    from sglang.srt.oft.oft_manager import OFTManager

    order = []
    event = mock.Mock()
    event.synchronize.side_effect = lambda: order.append("synchronize")
    manager = OFTManager.__new__(OFTManager)
    manager.pending_oft_load_events = {"adapter-a": event}
    manager.unload_adapter = mock.Mock(
        side_effect=lambda _ref: order.append("unload") or object()
    )
    ref = SimpleNamespace(adapter_id="adapter-a")

    manager.unload_oft_adapter(ref)

    assert order == ["synchronize", "unload"]
    assert "adapter-a" not in manager.pending_oft_load_events


def test_oft_manager_initializes_pending_load_store():
    from sglang.srt.oft.oft_manager import OFTManager

    manager = OFTManager.__new__(OFTManager)
    manager.init_oft_adapters()

    assert manager.pending_oft_load_events == {}
