"""Frozen H200 qualification inventory; explicit subsets never qualify the matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adapter_equivalence"

from .fixtures import MatrixCell, validate_matrix
from .schema import (
    SCHEMA_VERSION,
    BundleValidationError,
    CaseKey,
    _read_json_document,
    _require_exact_fields,
    _require_mapping,
    canonical_sha256,
)


@dataclass(frozen=True)
class QualificationScope:
    active_gpu_classes: tuple[str, ...]
    active_precisions: tuple[str, ...]
    active_architectures: tuple[str, ...]
    deferred_coverage: tuple[str, ...]

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self):
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}


CURRENT_SCOPE = QualificationScope(
    ("H200",),
    ("bf16", "fp8"),
    ("dense", "moe"),
    ("dense:nvfp4:B200", "moe:nvfp4:B200"),
)
PATH_NAMES = ("bundle", "completion", "jsonl", "stdout", "stderr", "blocked")
_FILENAMES = (
    "bundle.json",
    "complete.json",
    "events.jsonl",
    "stdout.log",
    "stderr.log",
    "blocked.json",
)


def exact_sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BundleValidationError("revision must be an exact lowercase 40-hex SHA")
    return value


def regular_artifact(path, *, dir_fd=None):
    info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BundleValidationError(
            f"artifact must be a regular single-link file: {path}"
        )
    return info


def _entry_identity(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def _file_identity(info):
    return (
        _entry_identity(info),
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class DirectoryAnchor:
    """No-follow, descriptor-relative traversal with observed-entry revalidation.

    This detects observed replacements, not arbitrary concurrent in-place
    mutation. A mutable filesystem is not an immutable snapshot.
    """

    def __init__(self, root):
        # Lexical normalization only: resolving symlinks would erase evidence.
        self.root = Path(os.path.abspath(root))
        self.directories = {}
        self.entries = {}
        self.names = {}
        try:
            self._directory(self.root)
            self.verify()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for fd, _ in reversed(tuple(self.directories.values())):
            try:
                os.close(fd)
            except OSError:
                pass
        self.directories.clear()

    def _directory(self, path):
        if path in self.directories:
            return self.directories[path][0]
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if path == Path("/"):
            fd = os.open("/", flags)
            before = os.fstat(fd)
        else:
            parent = self._directory(path.parent)
            before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise BundleValidationError("non-directory/symlink ancestor")
            fd = os.open(path.name, flags, dir_fd=parent)
        self.directories[path] = (fd, before)
        if _entry_identity(before) != _entry_identity(os.fstat(fd)):
            raise BundleValidationError("directory changed while opening")
        return fd

    def verify(self, *, inventory=False):
        for path, (fd, expected) in self.directories.items():
            actual = (
                os.fstat(fd)
                if path == Path("/")
                else os.stat(
                    path.name,
                    dir_fd=self.directories[path.parent][0],
                    follow_symlinks=False,
                )
            )
            if _entry_identity(actual) != _entry_identity(expected):
                raise BundleValidationError(
                    "directory entry changed during evidence read"
                )
        if inventory:
            for path, expected in self.names.items():
                if set(os.listdir(self.directories[path][0])) != expected:
                    raise BundleValidationError(
                        "inventory names changed during evidence read"
                    )
            for path, expected in self.entries.items():
                actual = os.stat(
                    path.name,
                    dir_fd=self.directories[path.parent][0],
                    follow_symlinks=False,
                )
                if _file_identity(actual) != _file_identity(expected):
                    raise BundleValidationError(
                        "inventory entry changed during evidence read"
                    )

    def inventory(self):
        def scan(path):
            fd = self._directory(path)
            with os.scandir(fd) as entries:
                names = {entry.name for entry in entries}
            self.names[path] = names
            for name in sorted(names):
                child = path / name
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    opened = os.fstat(self._directory(child))
                    if _entry_identity(info) != _entry_identity(opened):
                        raise BundleValidationError(
                            "directory changed during inventory"
                        )
                    scan(child)
                else:
                    checked = regular_artifact(name, dir_fd=fd)
                    if _file_identity(info) != _file_identity(checked):
                        raise BundleValidationError("file changed during inventory")
                    self.entries[child] = checked

        scan(self.root)
        self.verify(inventory=True)
        return frozenset(self.entries)

    def read(self, path):
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(self.root) or (
            self.names and path not in self.entries
        ):
            raise BundleValidationError("file is outside the anchored inventory")
        parent = self._directory(path.parent)
        before = regular_artifact(path.name, dir_fd=parent)
        if path in self.entries and _file_identity(before) != _file_identity(
            self.entries[path]
        ):
            raise BundleValidationError("file changed since inventory")
        self.verify()
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        with os.fdopen(fd, encoding="utf-8") as stream:
            opened = os.fstat(stream.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise BundleValidationError("artifact changed while opening")
            from .schema import _reject_duplicate_object_keys

            value = json.load(stream, object_pairs_hook=_reject_duplicate_object_keys)
            after = os.fstat(stream.fileno())
            entry = regular_artifact(path.name, dir_fd=parent)
            if _file_identity(before) != _file_identity(after) or _file_identity(
                before
            ) != _file_identity(entry):
                raise BundleValidationError("artifact changed while reading")
        self.verify()
        return value


def read_document(path, context, *, artifact=False, anchor=None):
    if anchor is not None:
        value = anchor.read(path)
    elif artifact:
        with DirectoryAnchor(Path(os.path.abspath(path)).parent) as guard:
            value = guard.read(path)
    else:
        value = _read_json_document(path, context)
    canonical_sha256(
        value
    )  # Reject nonfinite values anywhere, including unknown fields.
    return value


def outside_artifact_root(path, artifact_root):
    if Path(path).resolve().is_relative_to(Path(artifact_root).resolve()):
        raise BundleValidationError("publication output must be outside artifact_root")


def publish_document(path, value):
    """Exclusive atomic JSON publication; a published path is never rolled back."""
    destination = Path(path)
    payload = (
        json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if canonical_sha256(
            read_document(temporary, "publication")
        ) != canonical_sha256(value):
            raise BundleValidationError("publication readback mismatch")
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.link(temporary, destination)
        # The exclusive link is the commit point. Durability and private cleanup
        # after this point cannot turn a committed document into a failed publish.
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


@dataclass(frozen=True)
class ExpectedShard:
    case_id: str
    revision_kind: str
    repetition: int
    case_key: CaseKey
    gpu_class: str
    tp_size: int
    ep_size: int
    bundle_path: str
    completion_path: str
    jsonl_path: str
    stdout_path: str
    stderr_path: str
    blocked_path: str

    @classmethod
    def create(
        cls,
        *,
        case_id,
        revision_kind,
        repetition,
        git_sha,
        cell,
        mode,
        cuda_graph,
        artifact_root,
    ):
        root = artifact_root / case_id / revision_kind / f"rep-{repetition}"
        return cls(
            case_id,
            revision_kind,
            repetition,
            CaseKey(
                cell.model,
                cell.architecture,
                cell.precision,
                git_sha,
                mode,
                cuda_graph,
                "native-adapter-lifecycle-v2",
            ),
            cell.gpu,
            cell.tp,
            cell.ep,
            *(str(root / name) for name in _FILENAMES),
        )

    @property
    def identity(self):
        return self.case_id, self.revision_kind, self.repetition

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QualificationManifest:
    schema_version: int
    reference_sha: str
    candidate_sha: str
    artifact_root: str
    selected_case_ids: tuple[str, ...] | None
    qualifying: bool
    scope: QualificationScope
    shards: tuple[ExpectedShard, ...]
    manifest_hash: str

    def __post_init__(self):
        object.__setattr__(self, "shards", tuple(self.shards))
        if self.selected_case_ids is not None:
            object.__setattr__(self, "selected_case_ids", tuple(self.selected_case_ids))

    def _payload(self):
        return dict(
            schema_version=self.schema_version,
            reference_sha=self.reference_sha,
            candidate_sha=self.candidate_sha,
            artifact_root=self.artifact_root,
            selected_case_ids=(
                None if self.selected_case_ids is None else list(self.selected_case_ids)
            ),
            qualifying=self.qualifying,
            scope=self.scope.to_dict(),
            shards=[shard.to_dict() for shard in self.shards],
        )

    def validate(self):
        expected = build_manifest(
            validate_matrix(Path(__file__).with_name("matrix.json")),
            self.reference_sha,
            self.candidate_sha,
            Path(self.artifact_root),
            self.selected_case_ids,
        )
        if (
            canonical_sha256(self._payload()) != expected.manifest_hash
            or self.manifest_hash != expected.manifest_hash
        ):
            raise BundleValidationError(
                "qualification manifest differs from frozen inventory/hash"
            )

    def to_dict(self):
        self.validate()
        return dict(self._payload(), manifest_hash=self.manifest_hash)

    def write_json(self, path):
        outside_artifact_root(path, self.artifact_root)
        publish_document(path, self.to_dict())

    @classmethod
    def read_json(cls, path):
        data = _require_mapping(
            read_document(path, "QualificationManifest", artifact=True),
            "QualificationManifest",
        )
        _require_exact_fields(
            data, set(cls.__dataclass_fields__), "QualificationManifest"
        )
        scope = _require_mapping(data["scope"], "scope")
        _require_exact_fields(
            scope, set(QualificationScope.__dataclass_fields__), "scope"
        )
        if any(type(value) is not list for value in scope.values()):
            raise BundleValidationError("scope values must be arrays")
        if type(data["shards"]) is not list or (
            data["selected_case_ids"] is not None
            and type(data["selected_case_ids"]) is not list
        ):
            raise BundleValidationError("shards/selected_case_ids must be arrays")
        shards = []
        for raw in data["shards"]:
            raw = _require_mapping(raw, "shard")
            _require_exact_fields(raw, set(ExpectedShard.__dataclass_fields__), "shard")
            shards.append(
                ExpectedShard(**dict(raw, case_key=CaseKey.from_dict(raw["case_key"])))
            )
        manifest = cls(
            **dict(data, scope=QualificationScope(**scope), shards=tuple(shards))
        )
        manifest.validate()
        return manifest


def build_manifest(
    cells: tuple[MatrixCell, ...],
    reference_sha,
    candidate_sha,
    artifact_root,
    selected_case_ids=None,
):
    reference_sha, candidate_sha = exact_sha(reference_sha), exact_sha(candidate_sha)
    expected_cells = validate_matrix(Path(__file__).with_name("matrix.json"))
    if len(cells) != 4 or set(cells) != set(expected_cells):
        raise BundleValidationError(
            "current qualification requires dense/moe BF16/FP8 H200 cells exactly"
        )
    root = Path(artifact_root)
    if not root.is_absolute() or root != root.resolve():
        raise BundleValidationError("artifact_root must be an absolute canonical path")
    cases = {
        f"{c.id}.{m}.graph-{int(g)}"
        for c in cells
        for m in ("native_lora", "native_oft")
        for g in (False, True)
    }
    if selected_case_ids is not None:
        if (
            type(selected_case_ids) not in (tuple, list)
            or not selected_case_ids
            or any(type(case) is not str for case in selected_case_ids)
            or len(set(selected_case_ids)) != len(selected_case_ids)
            or not set(selected_case_ids) <= cases
        ):
            raise BundleValidationError("selection must contain unique known case IDs")
        selected_case_ids = tuple(sorted(selected_case_ids))
    shards = []
    for cell in expected_cells:
        for mode in ("native_lora", "native_oft"):
            for graph in (False, True):
                case_id = f"{cell.id}.{mode}.graph-{int(graph)}"
                if selected_case_ids is not None and case_id not in selected_case_ids:
                    continue
                for role, sha, count in (
                    ("source", reference_sha, 3),
                    ("candidate", candidate_sha, 1),
                ):
                    for repetition in range(count):
                        shards.append(
                            ExpectedShard.create(
                                case_id=case_id,
                                revision_kind=role,
                                repetition=repetition,
                                git_sha=sha,
                                cell=cell,
                                mode=mode,
                                cuda_graph=graph,
                                artifact_root=root,
                            )
                        )
    manifest = QualificationManifest(
        SCHEMA_VERSION,
        reference_sha,
        candidate_sha,
        str(root),
        selected_case_ids,
        selected_case_ids is None,
        CURRENT_SCOPE,
        tuple(shards),
        "",
    )
    object.__setattr__(manifest, "manifest_hash", canonical_sha256(manifest._payload()))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path, default=Path(__file__).with_name("matrix.json")
    )
    parser.add_argument("--reference-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--case-id", action="append", help="Explicit non-qualifying subset only"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Outside the shard artifact root"
    )
    args = parser.parse_args(argv)
    try:
        build_manifest(
            validate_matrix(args.matrix),
            args.reference_sha,
            args.candidate_sha,
            args.artifact_root,
            args.case_id,
        ).write_json(args.output)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
