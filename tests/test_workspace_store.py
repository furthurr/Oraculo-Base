import os
import asyncio
import stat
from pathlib import Path

import pytest

from oraculo.domain.models import ChatKind
from oraculo.domain.workspace import DocumentRef, WorkspaceError, WorkspaceScope
from oraculo.infrastructure.workspace import FileDocumentStore


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir(mode=0o700)
    return path


@pytest.fixture
def store(root: Path) -> FileDocumentStore:
    return FileDocumentStore(root)


USER = WorkspaceScope(ChatKind.PRIVATE, 123)
OTHER = WorkspaceScope(ChatKind.PRIVATE, 456)
GROUP = WorkspaceScope(ChatKind.GROUP, -123)


def test_root_and_references_are_validated(tmp_path: Path, root: Path) -> None:
    with pytest.raises(WorkspaceError):
        FileDocumentStore(tmp_path / "missing" / "workspace")
    root.chmod(0o755)
    with pytest.raises(WorkspaceError):
        FileDocumentStore(root)
    root.chmod(0o700)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(WorkspaceError):
        FileDocumentStore(link)
    for name in ("../bad.txt", "/tmp/a.txt", "temp/b.txt", ".hidden.md", "a..b.txt", "x.pdf", "a\\b.txt", "x" * 65 + ".md"):
        with pytest.raises(WorkspaceError):
            DocumentRef(name)
    with pytest.raises(WorkspaceError):
        WorkspaceScope(ChatKind.OTHER, 1)


@pytest.mark.asyncio
async def test_isolated_persistent_final_and_temp(store: FileDocumentStore, root: Path) -> None:
    ref = DocumentRef("notes.md")
    draft = DocumentRef("draft.txt", temporary=True)
    await store.create(USER, ref, "hola 🌎")
    await store.create(USER, draft, "temporal")
    assert (root / "users/123/notes.md").stat().st_mode & 0o777 == 0o600
    assert (root / "users/123/temp").stat().st_mode & 0o777 == 0o700
    assert await store.list(USER) == (ref, draft)
    assert await store.attachment(USER, draft) == b"temporal"
    assert (await FileDocumentStore(root).read(USER, ref))[0] == "hola 🌎"
    assert await store.list(OTHER) == ()
    with pytest.raises(WorkspaceError):
        await store.read(OTHER, ref)
    await store.create(GROUP, ref, "grupo")
    assert (await store.read(GROUP, ref))[0] == "grupo"
    assert (await store.read(USER, ref))[0] == "hola 🌎"


@pytest.mark.asyncio
async def test_create_is_exclusive_and_limits_content(store: FileDocumentStore, root: Path) -> None:
    ref = DocumentRef("a.md")
    await store.create(USER, ref, "original")
    with pytest.raises(WorkspaceError) as error:
        await store.create(USER, ref, "nuevo")
    assert error.value.reason == "already_exists"
    for text in ("a" * 3001, "界" * 2800, "\ud800"):
        with pytest.raises(WorkspaceError):
            await store.create(USER, DocumentRef("b.md"), text)
    assert (await store.read(USER, ref))[0] == "original"
    assert not (root / "users/123/b.md").exists()


@pytest.mark.asyncio
async def test_quota_and_read_validation(store: FileDocumentStore, root: Path) -> None:
    for i in range(100):
        await store.create(USER, DocumentRef(f"f{i}.txt", temporary=i % 2 == 0), "x")
    assert len(await store.list(USER)) == 100
    with pytest.raises(WorkspaceError) as error:
        await store.create(USER, DocumentRef("extra.md"), "x")
    assert error.value.reason == "quota"
    path = root / "users/123/f1.txt"
    path.write_bytes(b"\xff")
    with pytest.raises(WorkspaceError):
        await store.read(USER, DocumentRef("f1.txt"))
    path.write_bytes(b"x" * 8193)
    with pytest.raises(WorkspaceError):
        await store.attachment(USER, DocumentRef("f1.txt"))


@pytest.mark.asyncio
async def test_links_special_files_and_unexpected_directories_are_rejected(store: FileDocumentStore, root: Path) -> None:
    await store.create(USER, DocumentRef("ok.md"), "safe")
    folder = root / "users/123"
    (folder / "link.md").symlink_to(folder / "ok.md")
    with pytest.raises(WorkspaceError):
        await store.read(USER, DocumentRef("link.md"))
    (folder / "link.md").unlink()
    os.link(folder / "ok.md", folder / "hard.md")
    with pytest.raises(WorkspaceError):
        await store.attachment(USER, DocumentRef("ok.md"))
    (folder / "hard.md").unlink()
    (folder / "temp").symlink_to(root, target_is_directory=True)
    with pytest.raises(WorkspaceError):
        await store.create(USER, DocumentRef("draft.md", temporary=True), "x")


@pytest.mark.asyncio
async def test_version_conflicts_and_atomic_replace(store: FileDocumentStore, root: Path) -> None:
    ref = DocumentRef("plan.md")
    await store.create(USER, ref, "old")
    _, version = await store.read(USER, ref)
    await store.replace_if_version(USER, ref, "new", version)
    assert (await store.read(USER, ref))[0] == "new"
    with pytest.raises(WorkspaceError) as error:
        await store.replace_if_version(USER, ref, "lost", version)
    assert error.value.reason == "conflict"
    assert (await store.read(USER, ref))[0] == "new"
    assert sorted(p.name for p in (root / "users/123").iterdir()) == ["plan.md"]


@pytest.mark.asyncio
async def test_external_change_and_failed_staging_leave_original(store: FileDocumentStore, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = DocumentRef("plan.md")
    await store.create(USER, ref, "original")
    _, version = await store.read(USER, ref)
    path = root / "users/123/plan.md"
    path.write_text("external", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="conflict"):
        await store.replace_if_version(USER, ref, "proposal", version)

    def failed_sync(fd: int) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "fsync", failed_sync)
    with pytest.raises(WorkspaceError) as error:
        await store.replace_if_version(USER, ref, "proposal", (await store.read(USER, ref))[1])
    assert error.value.reason == "io_error"
    with pytest.raises(WorkspaceError):
        await store.create(USER, DocumentRef("new.txt"), "new")
    assert path.read_text(encoding="utf-8") == "external"
    assert sorted(p.name for p in path.parent.iterdir()) == ["plan.md"]


@pytest.mark.asyncio
async def test_failed_link_before_publication_cleans_stage(store: FileDocumentStore, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_link(src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int, follow_symlinks: bool) -> None:
        raise OSError("link unavailable")

    monkeypatch.setattr(os, "link", failed_link)
    with pytest.raises(WorkspaceError) as error:
        await store.create(USER, DocumentRef("new.txt"), "data")
    assert error.value.reason == "io_error"
    assert list((root / "users/123").iterdir()) == []


@pytest.mark.asyncio
async def test_concurrent_create_and_replaced_root(store: FileDocumentStore, root: Path, tmp_path: Path) -> None:
    ref = DocumentRef("same.txt")
    results = await asyncio.gather(
        store.create(USER, ref, "one"), store.create(USER, ref, "two"), return_exceptions=True
    )
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, WorkspaceError) and result.reason == "already_exists" for result in results) == 1
    original = tmp_path / "original"
    root.rename(original)
    root.symlink_to(original, target_is_directory=True)
    with pytest.raises(WorkspaceError):
        await store.attachment(USER, ref)


@pytest.mark.asyncio
async def test_rejects_public_existing_document(store: FileDocumentStore, root: Path) -> None:
    ref = DocumentRef("public.md")
    await store.create(USER, ref, "private")
    path = root / "users/123/public.md"
    path.chmod(0o644)
    with pytest.raises(WorkspaceError, match="unsafe_file"):
        await store.read(USER, ref)
    with pytest.raises(WorkspaceError, match="unsafe_file"):
        await store.list(USER)


@pytest.mark.asyncio
async def test_create_reports_unknown_durability_after_directory_sync_failure(store: FileDocumentStore, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_sync = os.fsync
    calls = 0

    def fail_second_sync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk unavailable")
        original_sync(fd)

    monkeypatch.setattr(os, "fsync", fail_second_sync)
    with pytest.raises(WorkspaceError) as error:
        await store.create(USER, DocumentRef("rolled.txt"), "data")
    assert error.value.reason == "durability_unknown"
    assert (await store.read(USER, DocumentRef("rolled.txt")))[0] == "data"
    assert sorted(p.name for p in (root / "users/123").iterdir()) == ["rolled.txt"]


@pytest.mark.asyncio
async def test_replace_reports_unknown_durability_after_directory_sync_failure(store: FileDocumentStore, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = DocumentRef("plan.md")
    await store.create(USER, ref, "original")
    _, version = await store.read(USER, ref)
    original_sync = os.fsync

    def fail_directory_sync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("disk unavailable")
        original_sync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    with pytest.raises(WorkspaceError) as error:
        await store.replace_if_version(USER, ref, "updated", version)
    assert error.value.reason == "durability_unknown"
    assert (await store.read(USER, ref))[0] == "updated"
    assert sorted(p.name for p in (root / "users/123").iterdir()) == ["plan.md"]
