import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "manual"))

from adapter_equivalence.fixtures import (
    AdapterFixture,
    FixtureValidationError,
    build_lora_fixture,
    build_oft_fixture,
    validate_matrix,
)
from adapter_equivalence.preflight import (
    PINNED_MODEL_REVISIONS,
    PreflightError,
    build_preflight_manifest,
    hash_checkpoint,
    load_prompt_manifest,
)


MANUAL_ROOT = Path(__file__).resolve().parents[3] / "manual" / "adapter_equivalence"
MATRIX_PATH = MANUAL_ROOT / "matrix.json"
PROMPTS_PATH = MANUAL_ROOT / "prompts.jsonl"

EXPECTED_CELLS = (
    (
        "qwen3-4b-bf16",
        "Qwen/Qwen3-4B-Instruct-2507",
        "dense",
        "bf16",
        "H100",
        1,
        1,
        None,
        None,
    ),
    (
        "qwen3-4b-fp8",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "dense",
        "fp8",
        "H100",
        1,
        1,
        "fp8",
        None,
    ),
    (
        "qwen3-4b-nvfp4",
        "OPENZEKA/Qwen3-4B-Instruct-2507-NVFP4",
        "dense",
        "nvfp4",
        "B200",
        1,
        1,
        "modelopt_fp4",
        None,
    ),
    (
        "qwen3-30b-a3b-bf16",
        "Qwen/Qwen3-30B-A3B",
        "moe",
        "bf16",
        "H100",
        4,
        4,
        None,
        None,
    ),
    (
        "qwen3-30b-a3b-fp8",
        "Qwen/Qwen3-30B-A3B-FP8",
        "moe",
        "fp8",
        "H100",
        4,
        4,
        "fp8",
        "triton",
    ),
    (
        "qwen3-30b-a3b-nvfp4",
        "nvidia/Qwen3-30B-A3B-NVFP4",
        "moe",
        "nvfp4",
        "B200",
        4,
        4,
        "modelopt_fp4",
        "flashinfer_cutlass",
    ),
)

EXPECTED_REVISIONS = {
    "qwen3-4b-bf16": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "qwen3-4b-fp8": "8591804019c8b22094c3b5b4454e0edc05dffc98",
    "qwen3-4b-nvfp4": "7009563e02c47b3ce728ecdc8cab2f0d9cd52ee4",
    "qwen3-30b-a3b-bf16": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "qwen3-30b-a3b-fp8": "d206ba732169f29bb77fbf80fc2c4b81d4d30782",
    "qwen3-30b-a3b-nvfp4": "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3",
}

EXPECTED_PROMPT_ID_HASHES = {
    "factual": "b9841adc036f9267cb4596f7e14bcc671d8272f687ddee8f3edf0449cec33b6d",
    "arithmetic": "e7f57264fbb9c3e707cc3ba64f81b0e47c354d007d29b7958c1e9febb9d34d86",
    "code": "c7b30ad99c68cdc76d29922537a14afe600c36b380c3695870c01d061461c330",
    "long-prefix": "9364fee68646386fa77421e3215a5c5a31e2cc1f45be535cb34a3e7124e33cb1",
    "uneven-mixed": "2cb42d40aaf46bb95962b46cd4f6ed8e03224adddfddbb98493e74547db46dbe",
    "graph-bucket": "12f32432e5baeba0e6b7f94049359cd05d1de937e304868d9a77f2f55b0cb3aa",
}


def _dense_shapes() -> dict[str, tuple[int, int]]:
    return {
        "model.layers.0.self_attn.q_proj": (128, 128),
        "model.layers.0.self_attn.o_proj": (128, 128),
        "model.layers.0.mlp.gate_proj": (128, 128),
        "model.layers.0.mlp.up_proj": (128, 128),
        "model.layers.0.mlp.down_proj": (128, 128),
    }


def _dense_lora_shapes() -> dict[str, tuple[int, int]]:
    return {
        **_dense_shapes(),
        "model.layers.0.self_attn.v_proj": (128, 128),
    }


def _moe_shapes() -> dict[str, tuple[int, int]]:
    return {
        "model.layers.0.self_attn.q_proj": (64, 64),
        "model.layers.0.self_attn.o_proj": (64, 64),
        "model.layers.0.mlp.experts.0.gate_proj": (32, 64),
        "model.layers.0.mlp.experts.0.up_proj": (32, 64),
        "model.layers.0.mlp.experts.0.down_proj": (64, 32),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_bytes(path: Path) -> dict[str, bytes]:
    return {
        name: (path / name).read_bytes()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "sha256.json",
        )
    }


def test_matrix_is_exactly_the_six_binding_cells() -> None:
    cells = validate_matrix(MATRIX_PATH)

    assert tuple(
        (
            cell.id,
            cell.model,
            cell.architecture,
            cell.precision,
            cell.gpu,
            cell.tp,
            cell.ep,
            cell.quantization,
            cell.moe_runner,
        )
        for cell in cells
    ) == EXPECTED_CELLS
    assert PINNED_MODEL_REVISIONS == EXPECTED_REVISIONS


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda rows: rows[0].update({"unknown": True}), "unknown fields"),
        (lambda rows: rows[0].update({"tp": True}), "tp must be an integer"),
        (lambda rows: rows[0].update({"gpu": "B200"}), "does not match"),
        (lambda rows: rows.reverse(), "matrix order"),
        (lambda rows: rows[1].update({"id": rows[0]["id"]}), "duplicate cell id"),
    ),
)
def test_matrix_validation_fails_closed(mutation, message: str) -> None:
    rows = json.loads(MATRIX_PATH.read_text())
    mutation(rows)

    with pytest.raises(FixtureValidationError, match=message):
        validate_matrix(rows)


def test_oft_generation_is_byte_stable_and_seeds_are_distinct(tmp_path: Path) -> None:
    first = build_oft_fixture(
        tmp_path / "first",
        adapter_id="adapter-a",
        architecture="dense",
        seed=1729,
        target_shapes=_dense_shapes(),
    )
    repeated = build_oft_fixture(
        tmp_path / "repeated",
        adapter_id="adapter-a",
        architecture="dense",
        seed=1729,
        target_shapes=_dense_shapes(),
    )
    second = build_oft_fixture(
        tmp_path / "second",
        adapter_id="adapter-b",
        architecture="dense",
        seed=2718,
        target_shapes=_dense_shapes(),
    )

    assert isinstance(first, AdapterFixture)
    assert _fixture_bytes(first.path) == _fixture_bytes(repeated.path)
    assert (
        _fixture_bytes(first.path)["adapter_model.safetensors"]
        != _fixture_bytes(second.path)["adapter_model.safetensors"]
    )


def test_oft_compact_weights_encode_identity_plus_skew_perturbation(
    tmp_path: Path,
) -> None:
    fixture = build_oft_fixture(
        tmp_path / "oft",
        adapter_id="adapter-a",
        architecture="dense",
        seed=1729,
        target_shapes=_dense_shapes(),
    )
    tensors = load_file(fixture.path / "adapter_model.safetensors")
    config = json.loads((fixture.path / "adapter_config.json").read_text())

    assert fixture.block_size == 128
    assert config["peft_type"] == "OFT"
    assert config["oft_block_size"] == 128
    assert config["target_modules"] == [
        "down_proj",
        "gate_proj",
        "o_proj",
        "q_proj",
        "up_proj",
    ]
    compact = tensors["base_model.model.model.layers.0.self_attn.q_proj.oft_R"]
    assert compact.shape == (1, 128 * 127 // 2)
    rows, columns = np.triu_indices(128, 1)
    skew = np.zeros((128, 128), dtype=np.float32)
    skew[rows, columns] = compact[0]
    skew -= skew.T
    identity_plus_perturbation = np.eye(128, dtype=np.float32) + skew
    np.testing.assert_array_equal(skew, -skew.T)
    np.testing.assert_array_equal(
        np.diag(identity_plus_perturbation), np.ones(128, dtype=np.float32)
    )
    assert np.count_nonzero(skew) > 0


def test_moe_oft_uses_block_32_and_expert_projection_keys(tmp_path: Path) -> None:
    fixture = build_oft_fixture(
        tmp_path / "moe",
        adapter_id="adapter-a",
        architecture="moe",
        seed=1729,
        target_shapes=_moe_shapes(),
    )
    tensors = load_file(fixture.path / "adapter_model.safetensors")

    assert fixture.block_size == 32
    assert any(".experts.0.gate_proj.oft_R" in name for name in tensors)
    assert any(".experts.0.up_proj.oft_R" in name for name in tensors)
    assert any(".experts.0.down_proj.oft_R" in name for name in tensors)


def test_lora_rank_alpha_shapes_and_byte_identity(tmp_path: Path) -> None:
    first = build_lora_fixture(
        tmp_path / "first",
        adapter_id="adapter-a",
        architecture="dense",
        seed=1729,
        target_shapes=_dense_lora_shapes(),
    )
    repeated = build_lora_fixture(
        tmp_path / "repeated",
        adapter_id="adapter-a",
        architecture="dense",
        seed=1729,
        target_shapes=_dense_lora_shapes(),
    )
    second = build_lora_fixture(
        tmp_path / "second",
        adapter_id="adapter-b",
        architecture="dense",
        seed=2718,
        target_shapes=_dense_lora_shapes(),
    )
    tensors = load_file(first.path / "adapter_model.safetensors")
    config = json.loads((first.path / "adapter_config.json").read_text())

    assert _fixture_bytes(first.path) == _fixture_bytes(repeated.path)
    assert (
        _fixture_bytes(first.path)["adapter_model.safetensors"]
        != _fixture_bytes(second.path)["adapter_model.safetensors"]
    )
    assert first.rank == 8
    assert first.alpha == 16
    assert config["r"] == 8
    assert config["lora_alpha"] == 16
    assert config["target_modules"] == [
        "down_proj",
        "gate_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert tensors[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ].shape == (8, 128)
    assert tensors[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    ].shape == (128, 8)
    assert tensors[
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"
    ].shape == (8, 128)
    assert tensors[
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
    ].shape == (128, 8)


@pytest.mark.parametrize(
    ("builder", "shapes", "message"),
    (
        (
            build_oft_fixture,
            {"model.layers.0.self_attn.q_proj": (128, 128)},
            "unresolved required targets",
        ),
        (
            build_lora_fixture,
            {
                **_dense_lora_shapes(),
                "model.layers.0.mlp.typo_proj": (128, 128),
            },
            "target",
        ),
        (
            build_lora_fixture,
            _dense_shapes(),
            "unresolved required targets: v_proj",
        ),
        (
            build_oft_fixture,
            {
                **_dense_shapes(),
                "model.layers.0.self_attn.q_proj": (96, 128),
            },
            "block size",
        ),
    ),
)
def test_fixture_generation_rejects_unresolved_or_invalid_targets(
    tmp_path: Path,
    builder,
    shapes: dict[str, tuple[int, int]],
    message: str,
) -> None:
    with pytest.raises(FixtureValidationError, match=message):
        builder(
            tmp_path / "invalid",
            adapter_id="adapter-a",
            architecture="dense",
            seed=1729,
            target_shapes=shapes,
        )


def test_fixture_hash_manifest_is_complete_and_excludes_itself(tmp_path: Path) -> None:
    fixture = build_oft_fixture(
        tmp_path / "fixture",
        adapter_id="adapter-a",
        architecture="dense",
        seed=1729,
        target_shapes=_dense_shapes(),
    )

    assert {path.name for path in fixture.path.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "sha256.json",
    }
    assert json.loads((fixture.path / "sha256.json").read_text()) == {
        "adapter_config.json": _sha256(fixture.path / "adapter_config.json"),
        "adapter_model.safetensors": _sha256(
            fixture.path / "adapter_model.safetensors"
        ),
    }


def test_fixture_generation_never_overwrites_an_existing_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "fixture"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep")

    with pytest.raises(FixtureValidationError, match="already exists"):
        build_oft_fixture(
            destination,
            adapter_id="adapter-a",
            architecture="dense",
            seed=1729,
            target_shapes=_dense_shapes(),
        )
    assert sentinel.read_text() == "keep"


def test_matrix_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "matrix.json"
    duplicate.write_text(
        '[{"id":"first","id":"second","model":"Qwen/example",'
        '"architecture":"dense","precision":"bf16","gpu":"H100",'
        '"tp":1,"ep":1}]\n'
    )

    with pytest.raises(FixtureValidationError, match="duplicate JSON key"):
        validate_matrix(duplicate)


def test_prompt_manifest_has_fixed_cases_and_temperature_zero_batches() -> None:
    manifest = load_prompt_manifest(PROMPTS_PATH)
    prompt_by_id = {prompt.id: prompt for prompt in manifest.prompts}

    assert manifest.tokenizer_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert (
        manifest.tokenizer_revision
        == "cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    assert tuple(prompt_by_id) == (
        "factual",
        "arithmetic",
        "code",
        "long-prefix",
        "uneven-mixed",
        "graph-bucket",
    )
    assert all(prompt.input_ids for prompt in manifest.prompts)
    assert len(prompt_by_id["long-prefix"].input_ids) > 256
    assert {
        prompt.id: hashlib.sha256(
            json.dumps(prompt.input_ids, separators=(",", ":")).encode()
        ).hexdigest()
        for prompt in manifest.prompts
    } == EXPECTED_PROMPT_ID_HASHES
    assert tuple(len(batch.requests) for batch in manifest.batches) == (1, 2, 8, 32)
    assert all(batch.temperature == 0 for batch in manifest.batches)

    request_ids = []
    for batch in manifest.batches:
        for request in batch.requests:
            request_ids.append(request.id)
            assert request.prompt_id in prompt_by_id
            assert request.input_ids == prompt_by_id[request.prompt_id].input_ids
    assert len(request_ids) == len(set(request_ids))


def _write_fake_checkpoint(snapshot: Path) -> None:
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3"}\n')
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 23},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}\n')
    (snapshot / "tokenizer_config.json").write_text('{"model_max_length":4096}\n')


def _write_unsharded_checkpoint(snapshot: Path) -> None:
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3"}\n')
    (snapshot / "model.safetensors").write_bytes(b"single weight file")
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}\n')
    (snapshot / "tokenizer_config.json").write_text(
        '{"model_max_length":4096}\n'
    )


def _write_six_fake_snapshots(tmp_path: Path) -> dict[str, Path]:
    snapshots = {}
    for cell_id, revision in EXPECTED_REVISIONS.items():
        snapshot = tmp_path / cell_id / revision
        _write_fake_checkpoint(snapshot)
        snapshots[cell_id] = snapshot
    return snapshots


def test_preflight_hashes_config_index_every_shard_and_tokenizer(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_fake_checkpoint(snapshot)

    result = hash_checkpoint(
        snapshot,
        model="Qwen/example",
        revision="0123456789abcdef0123456789abcdef01234567",
    )

    expected_names = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert result.path == snapshot.resolve()
    assert set(result.files) == expected_names
    assert result.files == {
        name: _sha256(snapshot / name) for name in sorted(expected_names)
    }
    assert result.layout == "indexed"
    assert result.index_hash == _sha256(
        snapshot / "model.safetensors.index.json"
    )
    assert len(result.checkpoint_hash) == 64
    assert len(result.tokenizer_hash) == 64
    assert result == hash_checkpoint(
        snapshot,
        model="Qwen/example",
        revision="0123456789abcdef0123456789abcdef01234567",
    )


def test_preflight_hashes_exactly_one_unsharded_safetensors(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_unsharded_checkpoint(snapshot)

    result = hash_checkpoint(
        snapshot,
        model="Qwen/example-unsharded",
        revision="0123456789abcdef0123456789abcdef01234567",
    )

    assert result.layout == "unsharded"
    assert result.index_hash is None
    assert set(result.files) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert result.files["model.safetensors"] == _sha256(
        snapshot / "model.safetensors"
    )


def test_preflight_manifest_resolves_and_hashes_all_six_cells(
    tmp_path: Path,
) -> None:
    snapshots = _write_six_fake_snapshots(tmp_path)

    manifest = build_preflight_manifest(
        MATRIX_PATH,
        PROMPTS_PATH,
        snapshots,
    )

    checkpoints = manifest["checkpoints"]
    assert isinstance(checkpoints, list)
    assert [checkpoint["id"] for checkpoint in checkpoints] == [
        cell[0] for cell in EXPECTED_CELLS
    ]
    assert [checkpoint["revision"] for checkpoint in checkpoints] == [
        EXPECTED_REVISIONS[cell[0]] for cell in EXPECTED_CELLS
    ]
    assert all(Path(checkpoint["path"]).is_absolute() for checkpoint in checkpoints)
    assert all(len(checkpoint["checkpoint_hash"]) == 64 for checkpoint in checkpoints)
    assert all(len(checkpoint["tokenizer_hash"]) == 64 for checkpoint in checkpoints)
    assert all(checkpoint["layout"] == "indexed" for checkpoint in checkpoints)
    assert all(len(checkpoint["index_hash"]) == 64 for checkpoint in checkpoints)
    assert manifest["matrix_sha256"] == _sha256(MATRIX_PATH)
    assert manifest["prompts_sha256"] == _sha256(PROMPTS_PATH)
    assert manifest["prompt_tokenizer"] == {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "files": {
            "tokenizer.json": (
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
            ),
            "tokenizer_config.json": (
                "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
            ),
        },
    }


def test_preflight_manifest_rejects_missing_and_mutable_cell_inputs(
    tmp_path: Path,
) -> None:
    snapshots = _write_six_fake_snapshots(tmp_path)
    missing = dict(snapshots)
    missing.pop("qwen3-4b-fp8")
    with pytest.raises(PreflightError, match="snapshot path keys mismatch"):
        build_preflight_manifest(
            MATRIX_PATH,
            PROMPTS_PATH,
            missing,
        )

    with pytest.raises(PreflightError, match="cannot override its pinned revision"):
        build_preflight_manifest(
            MATRIX_PATH,
            PROMPTS_PATH,
            snapshots,
            local_revisions={"qwen3-4b-nvfp4": "f" * 64},
        )


@pytest.mark.parametrize(
    ("bad_shard", "message"),
    (
        ("missing.safetensors", "missing shard"),
        ("../outside.safetensors", "unsafe shard"),
    ),
)
def test_preflight_rejects_partial_or_unsafe_checkpoints(
    tmp_path: Path, bad_shard: str, message: str
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_fake_checkpoint(snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layer.1"] = bad_shard
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n")

    with pytest.raises(PreflightError, match=message):
        hash_checkpoint(
            snapshot,
            model="Qwen/example",
            revision="0123456789abcdef0123456789abcdef01234567",
        )


def test_preflight_rejects_an_index_that_omits_a_numbered_shard(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_fake_checkpoint(snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("layer.1")
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
    (snapshot / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(PreflightError, match="incomplete shard sequence"):
        hash_checkpoint(
            snapshot,
            model="Qwen/example",
            revision="0123456789abcdef0123456789abcdef01234567",
        )


@pytest.mark.parametrize("mutation", ("extra", "numbered-without-index", "indexed-single"))
def test_preflight_rejects_ambiguous_or_mixed_weight_layouts(
    tmp_path: Path, mutation: str
) -> None:
    snapshot = tmp_path / mutation
    _write_unsharded_checkpoint(snapshot)
    if mutation == "extra":
        (snapshot / "other.safetensors").write_bytes(b"extra")
        message = "exactly one regular model.safetensors"
    elif mutation == "numbered-without-index":
        (snapshot / "model.safetensors").rename(
            snapshot / "model-00001-of-00001.safetensors"
        )
        message = "exactly one regular model.safetensors"
    else:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.0": "model.safetensors"}}) + "\n"
        )
        message = "contiguous numbered shards"

    with pytest.raises(PreflightError, match=message):
        hash_checkpoint(
            snapshot,
            model="Qwen/example",
            revision="0123456789abcdef0123456789abcdef01234567",
        )
