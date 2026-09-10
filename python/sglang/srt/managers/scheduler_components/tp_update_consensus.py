from typing import Any, Callable, List, Optional, Sequence, Tuple

TpUpdateResult = Tuple[bool, str, Optional[str]]
TpAdapterUpdateResult = Tuple[bool, str, Optional[str], Optional[str]]


def gather_tp_update_result(
    *,
    distributed: Any,
    group: Any,
    success: bool,
    message: str,
    version: Optional[str],
) -> TpUpdateResult:
    """Make every TP rank observe the same control-plane update result."""
    local_result = (success, message, version)
    world_size = distributed.get_world_size(group=group)
    results: List[Optional[TpUpdateResult]] = [None] * world_size
    distributed.all_gather_object(results, local_result, group=group)

    if not results or any(result is None for result in results):
        return False, "TP consensus is missing rank results", None

    failures = [
        f"TP rank {rank}: {result[1]}"
        for rank, result in enumerate(results)
        if result is not None and not result[0]
    ]
    if failures:
        return False, " | ".join(failures), None

    versions = [result[2] for result in results if result is not None]
    if len(set(versions)) != 1:
        version_details = ", ".join(
            f"rank {rank}={result[2]!r}"
            for rank, result in enumerate(results)
            if result is not None
        )
        return (
            False,
            f"TP ranks reported different versions: {version_details}",
            None,
        )

    return success, message, versions[0]


def run_tp_adapter_update(
    *,
    distributed: Any,
    group: Any,
    version: Optional[str],
    activate_immediately: bool,
    stage: Callable[[], Tuple[bool, str]],
    activate: Callable[[], Tuple[bool, str]],
) -> TpAdapterUpdateResult:
    """Stage an adapter and activate it only after unanimous TP agreement."""
    try:
        local_success, local_message = stage()
    except Exception as error:
        local_success = False
        local_message = f"Failed to stage adapter: {error}"

    stage_success, message, staged_version = gather_tp_update_result(
        distributed=distributed,
        group=group,
        success=local_success,
        message=local_message,
        version=version if local_success else None,
    )
    if not stage_success or not activate_immediately:
        return stage_success, message, staged_version, None

    activation_success, message, active_version = run_tp_adapter_activation(
        distributed=distributed,
        group=group,
        version=version,
        activate=activate,
    )
    return activation_success, message, staged_version, active_version


def run_tp_adapter_activation(
    *,
    distributed: Any,
    group: Any,
    version: Optional[str],
    activate: Callable[[], Tuple[bool, str]],
) -> TpUpdateResult:
    """Activate an adapter and require every TP rank to report success."""
    try:
        local_success, local_message = activate()
    except Exception as error:
        local_success = False
        local_message = f"Failed to activate adapter: {error}"

    return gather_tp_update_result(
        distributed=distributed,
        group=group,
        success=local_success,
        message=local_message,
        version=version if local_success else None,
    )


def run_tp_adapter_stage_discard(
    *,
    distributed: Any,
    group: Any,
    discard: Callable[[], Tuple[bool, str]],
) -> TpUpdateResult:
    """Every rank discards locally, including ranks whose stage never existed."""
    try:
        success, message = discard()
    except Exception as error:
        success, message = False, f"Failed to discard adapter stage: {error}"
    try:
        return gather_tp_update_result(
            distributed=distributed,
            group=group,
            success=success,
            message=message,
            version=None,
        )
    except Exception as error:
        return False, f"Adapter stage discard consensus failed: {error}", None


def run_tp_adapter_unload(
    *,
    distributed: Any,
    groups: Sequence[Any],
    unload: Callable[[], Any],
    output_type: Any,
) -> Any:
    """Propagate cleanup failures across the groups carrying this control request."""
    try:
        result = unload()
    except Exception as error:
        result = output_type(success=False, error_message=str(error))

    # Attention TP followed by attention CP spans one attention-DP domain.
    # Continue through every group after a failure so the other dimension also
    # observes it. The tokenizer merges replies from independent DP domains.
    for group in groups:
        success, message, _ = gather_tp_update_result(
            distributed=distributed,
            group=group,
            success=result.success,
            message=result.error_message or "",
            version=None,
        )
        if not success:
            result = output_type(
                success=False,
                error_message=message,
                loaded_adapters=result.loaded_adapters,
            )
    return result


def run_tp_oft_load(
    *,
    distributed: Any,
    groups: Sequence[Any],
    load: Callable[[], Any],
    get_loaded_ref: Callable[[], Any],
    expected_ref: Any,
    output_type: Any,
) -> Any:
    """Publish native OFT loads only after every worker verifies installed state."""
    try:
        result = load()
        if result.success:
            installed = get_loaded_ref()
            fields = ("adapter_name", "adapter_id", "adapter_version")
            actual = tuple(getattr(installed, field, None) for field in fields)
            expected = tuple(getattr(expected_ref, field) for field in fields)
            if actual != expected:
                result = output_type(
                    success=False,
                    error_message=(
                        f"Installed OFT identity {actual!r} != requested {expected!r}"
                    ),
                    inconsistent_update=True,
                )
    except Exception as error:
        # An exception can follow mutation; no safe rollback was confirmed.
        result = output_type(
            success=False, error_message=str(error), inconsistent_update=True
        )

    # Use the same TP/CP domains as control broadcast and unload. A failed
    # first dimension must still participate in subsequent dimensions.
    for group in groups:
        local = (
            result.success,
            result.error_message or "",
            result.previous_adapter_preserved,
            result.inconsistent_update,
        )
        try:
            replies = [None] * distributed.get_world_size(group=group)
            distributed.all_gather_object(replies, local, group=group)
            if not replies or any(reply is None for reply in replies):
                raise RuntimeError("OFT load consensus is missing rank results")
            failures = [
                f"TP rank {rank}: {reply[1]}"
                for rank, reply in enumerate(replies)
                if not reply[0]
            ]
            inconsistent = any(reply[3] for reply in replies) or (
                bool(failures) and any(reply[0] for reply in replies)
            )
            result = output_type(
                success=not failures and not inconsistent,
                error_message=" | ".join(failures),
                loaded_adapters=result.loaded_adapters,
                previous_adapter_preserved=(
                    not inconsistent and all(not r[0] and r[2] for r in replies)
                ),
                inconsistent_update=inconsistent,
            )
        except Exception as error:
            result = output_type(
                success=False,
                error_message=f"OFT load consensus failed: {error}",
                inconsistent_update=True,
            )
    return result
