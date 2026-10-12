"""Catálogo inmutable de skills declarativas cargadas desde archivos locales."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import yaml

from oraculo.domain.models import Skill, SkillCatalogSnapshot

logger = logging.getLogger(__name__)

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ALLOWED_KEYS = frozenset(
    {"name", "description", "license", "compatibility", "metadata"}
)


class _InvalidSkill(Exception):
    def __init__(self, cause: str) -> None:
        self.cause = cause
        super().__init__(cause)


class FileSkillCatalog:
    """Lee `<root>/*/SKILL.md` una vez y conserva el resultado en memoria."""

    def __init__(self, root: str | Path, *, max_chars: int) -> None:
        if max_chars < 1:
            raise ValueError("max_chars debe ser mayor que cero")
        self._root = Path(root).expanduser().resolve()
        self._max_chars = max_chars
        self._snapshot = self._load_snapshot()

    @property
    def snapshot(self) -> SkillCatalogSnapshot:
        """Devuelve el mismo snapshot sin volver a consultar el filesystem."""
        return self._snapshot

    def _load_snapshot(self) -> SkillCatalogSnapshot:
        if not self._root.is_dir():
            logger.info("Skill catalog loaded=0 skipped=0")
            return SkillCatalogSnapshot()

        loaded: list[Skill] = []
        skipped = 0
        try:
            entries = tuple(self._root.iterdir())
        except OSError:
            logger.warning("Skill catalog unavailable cause=root_unreadable")
            logger.info("Skill catalog loaded=0 skipped=0")
            return SkillCatalogSnapshot()

        for entry in entries:
            if not entry.is_dir():
                continue
            try:
                skill = self._load_entry(entry)
            except _InvalidSkill as exc:
                skipped += 1
                logger.warning("Skill omitted cause=%s", exc.cause)
                continue
            if skill is not None:
                loaded.append(skill)

        counts: dict[str, int] = {}
        for skill in loaded:
            counts[skill.name] = counts.get(skill.name, 0) + 1
        collisions = {name for name, count in counts.items() if count > 1}
        if collisions:
            for name in sorted(collisions):
                logger.warning("Skill omitted cause=duplicate skill_name=%s", name)
            kept = [skill for skill in loaded if skill.name not in collisions]
            skipped += len(loaded) - len(kept)
            loaded = kept

        snapshot = SkillCatalogSnapshot(tuple(loaded))
        logger.info(
            "Skill catalog loaded=%s skipped=%s",
            len(snapshot.skills),
            skipped,
        )
        return snapshot

    def _load_entry(self, entry: Path) -> Skill | None:
        try:
            directory = entry.resolve(strict=True)
        except OSError as exc:
            raise _InvalidSkill("path_unresolvable") from exc
        if not self._is_within_root(directory):
            raise _InvalidSkill("path_escape")

        candidate = directory / "SKILL.md"
        if not candidate.exists():
            return None
        try:
            resolved_file = candidate.resolve(strict=True)
        except OSError as exc:
            raise _InvalidSkill("path_unresolvable") from exc
        if not self._is_within_root(resolved_file) or resolved_file.parent != directory:
            raise _InvalidSkill("path_escape")
        if not resolved_file.is_file():
            raise _InvalidSkill("not_regular_file")

        try:
            raw = resolved_file.read_bytes()
        except OSError as exc:
            raise _InvalidSkill("file_unreadable") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _InvalidSkill("invalid_utf8") from exc

        metadata, instructions = self._parse_document(text)
        name = self._required_text(metadata, "name")
        description = self._required_text(metadata, "description")
        if len(name) > 64 or _SLUG_PATTERN.fullmatch(name) is None:
            raise _InvalidSkill("invalid_name")
        if name != entry.name:
            raise _InvalidSkill("directory_mismatch")
        if len(description) > 500 or "\n" in description or "\r" in description:
            raise _InvalidSkill("invalid_description")
        self._validate_optional_metadata(metadata)
        if not instructions:
            raise _InvalidSkill("empty_body")
        if len(instructions) > self._max_chars:
            raise _InvalidSkill("body_too_large")

        license_value = metadata.get("license")
        compatibility_value = metadata.get("compatibility")
        extra_metadata = metadata.get("metadata")
        metadata_items = (
            tuple(sorted(extra_metadata.items()))
            if isinstance(extra_metadata, dict)
            else ()
        )

        return Skill(
            name=name,
            description=description,
            instructions=instructions,
            content_hash=hashlib.sha256(raw).hexdigest(),
            license=license_value if isinstance(license_value, str) else None,
            compatibility=(
                compatibility_value
                if isinstance(compatibility_value, str)
                else None
            ),
            metadata=metadata_items,
        )

    def _parse_document(self, text: str) -> tuple[dict[object, object], str]:
        lines = text.splitlines()
        if not lines or lines[0] != "---":
            raise _InvalidSkill("missing_frontmatter")
        try:
            closing = lines.index("---", 1)
        except ValueError as exc:
            raise _InvalidSkill("missing_frontmatter_end") from exc
        yaml_source = "\n".join(lines[1:closing])
        try:
            parsed = yaml.safe_load(yaml_source)
        except yaml.YAMLError as exc:
            raise _InvalidSkill("invalid_yaml") from exc
        if not isinstance(parsed, dict):
            raise _InvalidSkill("frontmatter_not_mapping")
        if any(not isinstance(key, str) for key in parsed):
            raise _InvalidSkill("invalid_key_type")
        if not set(parsed).issubset(_ALLOWED_KEYS):
            raise _InvalidSkill("unknown_key")
        return parsed, "\n".join(lines[closing + 1 :]).strip()

    @staticmethod
    def _required_text(metadata: dict[object, object], key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _InvalidSkill(f"invalid_{key}")
        if value != value.strip():
            raise _InvalidSkill(f"invalid_{key}")
        return value

    @staticmethod
    def _validate_optional_metadata(metadata: dict[object, object]) -> None:
        for key in ("license", "compatibility"):
            value = metadata.get(key)
            if value is not None and not isinstance(value, str):
                raise _InvalidSkill(f"invalid_{key}")
        optional = metadata.get("metadata")
        if optional is None:
            return
        if not isinstance(optional, dict):
            raise _InvalidSkill("invalid_metadata")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in optional.items()
        ):
            raise _InvalidSkill("invalid_metadata")

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self._root)
        except ValueError:
            return False
        return path != self._root
