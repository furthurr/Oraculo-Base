"""Preparación de catálogo y skills explícitas para un turno de conversación."""

from __future__ import annotations

import logging
import re

from oraculo.domain.errors import SkillLimitError
from oraculo.domain.models import (
    Skill,
    SkillCatalogSnapshot,
    SkillPreparation,
    SkillTurnContext,
)

logger = logging.getLogger(__name__)

_SKILL_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])\$([a-z0-9]+(?:-[a-z0-9]+)*)(?![A-Za-z0-9_-])"
)
_CATALOG_OPEN = "<skills_catalog>\n"
_CATALOG_CLOSE = "\n</skills_catalog>"
_TRUNCATED = "- descriptions truncated"


class SkillService:
    """Construye contexto efímero únicamente desde un snapshot en memoria."""

    def __init__(
        self,
        snapshot: SkillCatalogSnapshot,
        *,
        autoload_enabled: bool,
        catalog_max_chars: int,
        max_context_chars: int,
        max_per_turn: int,
        max_rounds: int,
    ) -> None:
        if min(catalog_max_chars, max_context_chars, max_per_turn, max_rounds) < 1:
            raise SkillLimitError("los límites deben ser mayores que cero")
        self._snapshot = snapshot
        self._autoload_enabled = autoload_enabled
        self._max_context_chars = max_context_chars
        self._max_per_turn = max_per_turn
        self._max_rounds = max_rounds
        self._catalog_text = self._build_catalog(
            min(catalog_max_chars, max_context_chars)
        )

    @property
    def snapshot(self) -> SkillCatalogSnapshot:
        return self._snapshot

    def prepare(self, prompt: str) -> SkillPreparation:
        """Reconoce invocaciones explícitas sin modificar el texto persistible."""
        selected: list[Skill] = []
        seen: set[str] = set()
        for match in _SKILL_TOKEN.finditer(prompt):
            name = match.group(1)
            skill = self._snapshot.find(name)
            if skill is None or name in seen:
                continue
            seen.add(name)
            selected.append(skill)
            if len(selected) > self._max_per_turn:
                raise SkillLimitError("demasiadas skills explícitas")

        context_size = len(self._catalog_text) + sum(
            self._explicit_block_size(skill) for skill in selected
        )
        if context_size > self._max_context_chars:
            raise SkillLimitError("contenido explícito demasiado grande")

        for skill in selected:
            logger.info("Skill selected skill_name=%s mode=explicit", skill.name)
        context = SkillTurnContext(
            catalog_text=self._catalog_text,
            explicit_skills=tuple(selected),
            loadable_skills=(
                self._snapshot.skills if self._autoload_enabled else ()
            ),
            autoload_enabled=self._autoload_enabled and bool(self._snapshot.skills),
            max_rounds=self._max_rounds,
            max_context_chars=self._max_context_chars,
        )
        return SkillPreparation(prompt=prompt, context=context)

    @staticmethod
    def _explicit_block_size(skill: Skill) -> int:
        return len(f'<skill name="{skill.name}">\n{skill.instructions}\n</skill>')

    def _build_catalog(self, budget: int) -> str:
        skills = self._snapshot.skills
        if not skills:
            return ""

        full = self._render_catalog(
            tuple(f"- ${skill.name}: {skill.description}" for skill in skills)
        )
        if len(full) <= budget:
            return full

        id_lines = tuple(f"- ${skill.name}" for skill in skills)
        minimum = self._render_catalog((*id_lines, _TRUNCATED))
        if len(minimum) <= budget:
            maximum = max(len(skill.description) for skill in skills)
            low, high = 0, maximum
            while low < high:
                middle = (low + high + 1) // 2
                candidate = self._catalog_with_description_cap(middle)
                if len(candidate) <= budget:
                    low = middle
                else:
                    high = middle - 1
            return self._catalog_with_description_cap(low)

        for count in range(len(skills), -1, -1):
            omitted = len(skills) - count
            lines = tuple(f"- ${skill.name}" for skill in skills[:count])
            marker = f"- {omitted} skill ids omitted"
            candidate = self._render_catalog((*lines, marker))
            if len(candidate) <= budget:
                return candidate

        raise SkillLimitError("presupuesto insuficiente para delimitar el catálogo")

    def _catalog_with_description_cap(self, cap: int) -> str:
        lines: list[str] = []
        for skill in self._snapshot.skills:
            if cap == 0:
                lines.append(f"- ${skill.name}")
                continue
            description = skill.description
            if len(description) > cap:
                description = description[: max(0, cap - 1)] + "…"
            lines.append(f"- ${skill.name}: {description}")
        lines.append(_TRUNCATED)
        return self._render_catalog(tuple(lines))

    @staticmethod
    def _render_catalog(lines: tuple[str, ...]) -> str:
        return _CATALOG_OPEN + "\n".join(lines) + _CATALOG_CLOSE
