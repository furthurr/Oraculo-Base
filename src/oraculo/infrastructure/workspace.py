"""Persistencia de documentos con descriptores anclados y publicación atómica."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from oraculo.domain.models import ChatKind
from oraculo.domain.workspace import DocumentRef, DocumentVersion, WorkspaceError, WorkspaceScope

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_MAX_BYTES = 8192
_MAX_CHARS = 3000
_T = TypeVar("_T")


def _private_dir(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise WorkspaceError("unsafe_directory")


def _open_dir(parent: int, name: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
    try:
        _private_dir(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


class FileDocumentStore:
    """Único adaptador con acceso al disco; la autorización corresponde a aplicación."""

    def __init__(self, root: Path) -> None:
        supplied = Path(root).expanduser()
        if ".." in supplied.parts or supplied.name != "workspace":
            raise WorkspaceError("invalid_root")
        path = supplied.absolute()
        try:
            with self._open_root_path(path) as fd:
                _private_dir(fd)
        except OSError as exc:
            raise WorkspaceError("invalid_root") from exc
        self._root = path
        self._lock = threading.RLock()

    @staticmethod
    @contextmanager
    def _open_root_path(path: Path) -> Iterator[int]:
        fd = os.open("/", _DIR_FLAGS)
        try:
            for component in path.parts[1:]:
                next_fd = os.open(component, _DIR_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def _directory(self, scope: WorkspaceScope, temporary: bool = False, *, create: bool = False) -> Iterator[int]:
        if not isinstance(scope, WorkspaceScope):
            raise WorkspaceError("invalid_scope")
        with self._open_root_path(self._root) as root:
            _private_dir(root)
            section = _open_dir(root, "users" if scope.kind is ChatKind.PRIVATE else "groups", create=create)
            try:
                own = _open_dir(section, str(scope.id), create=create)
                try:
                    if temporary:
                        temp = _open_dir(own, "temp", create=create)
                        try:
                            yield temp
                        finally:
                            os.close(temp)
                    else:
                        yield own
                finally:
                    os.close(own)
            finally:
                os.close(section)

    @staticmethod
    def _check_ref(ref: DocumentRef) -> None:
        if not isinstance(ref, DocumentRef):
            raise WorkspaceError("invalid_reference")
        # Recheck against forged dataclass instances.
        ref.__post_init__()

    @staticmethod
    def _read(fd: int, name: str) -> tuple[str, bytes, DocumentVersion]:
        handle = os.open(name, _FILE_FLAGS, dir_fd=fd)
        try:
            before = os.fstat(handle)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_uid != os.getuid() or before.st_mode & 0o077):
                raise WorkspaceError("unsafe_file")
            if before.st_size > _MAX_BYTES:
                raise WorkspaceError("quota")
            raw = bytearray()
            while len(raw) <= _MAX_BYTES:
                piece = os.read(handle, _MAX_BYTES + 1 - len(raw))
                if not piece:
                    break
                raw.extend(piece)
            if len(raw) > _MAX_BYTES:
                raise WorkspaceError("quota")
            after = os.fstat(handle)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ) or len(raw) != after.st_size or after.st_nlink != 1:
                raise WorkspaceError("conflict")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspaceError("invalid_format") from exc
            if len(text) > _MAX_CHARS:
                raise WorkspaceError("quota")
            return text, bytes(raw), DocumentVersion(after.st_dev, after.st_ino, hashlib.sha256(raw).hexdigest())
        finally:
            os.close(handle)

    @staticmethod
    def _encode(content: str) -> bytes:
        if not isinstance(content, str):
            raise WorkspaceError("invalid_format")
        if len(content) > _MAX_CHARS:
            raise WorkspaceError("quota")
        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkspaceError("invalid_format") from exc
        if len(raw) > _MAX_BYTES:
            raise WorkspaceError("quota")
        return raw

    @staticmethod
    def _names(fd: int, *, allow_temp: bool = False) -> list[str]:
        names: list[str] = []
        for name in os.listdir(fd):
            if name == "temp" and allow_temp:
                continue
            DocumentRef(name)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid != os.getuid() or info.st_mode & 0o077):
                raise WorkspaceError("unsafe_file")
            names.append(name)
        return sorted(names)

    def _list(self, scope: WorkspaceScope) -> tuple[DocumentRef, ...]:
        with self._lock:
            try:
                with self._directory(scope) as fd:
                    main = [DocumentRef(name) for name in self._names(fd, allow_temp=True)]
                    try:
                        temp = _open_dir(fd, "temp", create=False)
                    except FileNotFoundError:
                        if len(main) > 100:
                            raise WorkspaceError("quota")
                        return tuple(main)
                    try:
                        drafts = [DocumentRef(name, temporary=True) for name in self._names(temp)]
                    finally:
                        os.close(temp)
                    if len(main) + len(drafts) > 100:
                        raise WorkspaceError("quota")
                    return tuple(main + drafts)
            except FileNotFoundError:
                return ()

    def _read_document(self, scope: WorkspaceScope, ref: DocumentRef) -> tuple[str, DocumentVersion]:
        self._check_ref(ref)
        with self._directory(scope, ref.temporary) as fd:
            text, _, version = self._read(fd, ref.name)
            return text, version

    def _attachment(self, scope: WorkspaceScope, ref: DocumentRef) -> bytes:
        self._check_ref(ref)
        with self._directory(scope, ref.temporary) as fd:
            _, raw, _ = self._read(fd, ref.name)
            return raw

    def _count(self, scope: WorkspaceScope) -> int:
        with self._directory(scope) as fd:
            main = self._names(fd, allow_temp=True)
            try:
                temp = _open_dir(fd, "temp", create=False)
            except FileNotFoundError:
                return len(main)
            try:
                return len(main) + len(self._names(temp))
            finally:
                os.close(temp)

    @staticmethod
    def _stage(fd: int, raw: bytes) -> str:
        name = ".stage-" + secrets.token_hex(16)
        handle = os.open(name, _WRITE_FLAGS, 0o600, dir_fd=fd)
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            os.unlink(name, dir_fd=fd)
            raise
        return name

    def _create(self, scope: WorkspaceScope, ref: DocumentRef, content: str) -> None:
        self._check_ref(ref)
        raw = self._encode(content)
        with self._lock:
            with self._directory(scope, create=True):
                if self._count(scope) >= 100:
                    raise WorkspaceError("quota")
            with self._directory(scope, ref.temporary, create=True) as fd:
                staged = self._stage(fd, raw)
                published = False
                try:
                    os.link(staged, ref.name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
                    published = True
                    os.unlink(staged, dir_fd=fd)
                    os.fsync(fd)
                except OSError as exc:
                    if published:
                        raise WorkspaceError("durability_unknown") from exc
                    raise
                finally:
                    try:
                        os.unlink(staged, dir_fd=fd)
                    except FileNotFoundError:
                        pass

    def _replace(self, scope: WorkspaceScope, ref: DocumentRef, content: str, version: DocumentVersion) -> None:
        self._check_ref(ref)
        if not isinstance(version, DocumentVersion):
            raise WorkspaceError("invalid_version")
        raw = self._encode(content)
        with self._lock:
            with self._directory(scope, ref.temporary) as fd:
                _, _, current = self._read(fd, ref.name)
                if current != version:
                    raise WorkspaceError("conflict")
                staged = self._stage(fd, raw)
                try:
                    os.replace(staged, ref.name, src_dir_fd=fd, dst_dir_fd=fd)
                    try:
                        os.fsync(fd)
                    except OSError as exc:
                        raise WorkspaceError("durability_unknown") from exc
                finally:
                    try:
                        os.unlink(staged, dir_fd=fd)
                    except FileNotFoundError:
                        pass

    @staticmethod
    def _safe_call(operation: Callable[..., _T], *args: object) -> _T:
        try:
            return operation(*args)
        except WorkspaceError:
            raise
        except FileNotFoundError as exc:
            raise WorkspaceError("not_found") from exc
        except FileExistsError as exc:
            raise WorkspaceError("already_exists") from exc
        except OSError as exc:
            raise WorkspaceError("io_error") from exc

    async def list(self, scope: WorkspaceScope) -> tuple[DocumentRef, ...]:
        return await asyncio.to_thread(self._safe_call, self._list, scope)

    async def read(self, scope: WorkspaceScope, ref: DocumentRef) -> tuple[str, DocumentVersion]:
        return await asyncio.to_thread(self._safe_call, self._read_document, scope, ref)

    async def create(self, scope: WorkspaceScope, ref: DocumentRef, content: str) -> None:
        await asyncio.to_thread(self._safe_call, self._create, scope, ref, content)

    async def replace_if_version(self, scope: WorkspaceScope, ref: DocumentRef, content: str, version: DocumentVersion) -> None:
        await asyncio.to_thread(self._safe_call, self._replace, scope, ref, content, version)

    async def attachment(self, scope: WorkspaceScope, ref: DocumentRef) -> bytes:
        return await asyncio.to_thread(self._safe_call, self._attachment, scope, ref)
