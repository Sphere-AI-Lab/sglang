def test_tensor_load_request_uses_per_rank_payload_and_nonreloadable_ref():
    from sglang.srt.oft.io_types import LoadOFTAdapterFromTensorsReqInput

    request = LoadOFTAdapterFromTensorsReqInput(
        adapter_name="policy",
        config_dict={"oft_block_size": 32},
        serialized_named_tensors=[b"rank-0", b"rank-1"],
        pinned=True,
        load_format="safetensors",
        upsert=True,
        adapter_id="adapter-id",
        adapter_version=7,
    )

    assert request.serialized_named_tensors == [b"rank-0", b"rank-1"]
    assert request.load_format == "safetensors"
    assert request.upsert is True
    assert request.to_ref().adapter_path == "__tensor__"
    assert request.to_ref().pinned is True
    assert request.to_ref().adapter_version == 7
    assert request.to_ref().reloadable is False


def test_distributed_load_request_carries_tensor_metadata_and_nonreloadable_ref():
    import sglang.srt.oft.io_types as io_types

    request_cls = getattr(io_types, "LoadOFTAdapterFromDistributedReqInput")
    request = request_cls(
        adapter_name="policy",
        config_dict={"oft_block_size": 32},
        names=["layer.weight"],
        dtypes=["float16"],
        shapes=[[32, 32]],
        upsert=True,
        adapter_id="adapter-id",
        adapter_version=9,
    )

    assert request.group_name == "weight_update_group"
    assert request.names == ["layer.weight"]
    assert request.dtypes == ["float16"]
    assert request.shapes == [[32, 32]]
    assert request.upsert is True
    assert request.to_ref().adapter_path == "__distributed__"
    assert request.to_ref().adapter_version == 9
    assert request.to_ref().reloadable is False


def test_oft_update_output_accepts_distributed_output_alias():
    import sglang.srt.oft.io_types as io_types

    output_cls = getattr(io_types, "LoadOFTAdapterFromDistributedReqOutput")
    output = output_cls(
        success=True,
        loaded_adapters={"policy": "adapter-id"},
    )

    assert output.success is True
    assert output.loaded_adapters == {"policy": "adapter-id"}
