"""Carga y validación de la configuración desde variables de entorno."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from oraculo.domain.errors import ConfigError
from oraculo.domain.models import AppConfig
from oraculo.infrastructure.mimo import MIMO_BASE_URL

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MIMO_MODEL = "xiaomi/mimo-v2.6-flash"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_TOKENS = 1024
# Por defecto sin thinking: respuestas rápidas y baratas para un chat casual.
DEFAULT_THINKING = False
# Solo se aplica cuando el thinking está habilitado.
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_AI_PROVIDER = "mimo"
# Segundos mínimos entre ediciones del mensaje en streaming de Pi (flood control).
DEFAULT_PI_STREAM_EDIT_INTERVAL_S = 1.0

_THINKING_CHOICES = ("enabled", "disabled")
_REASONING_EFFORT_CHOICES = ("low", "high", "max")

# Instrucción de sistema enviada a DeepSeek: fija el idioma de las respuestas.
DEFAULT_SYSTEM_PROMPT = (
    "Responde siempre en español, de forma clara y natural. "
    "Solo responde en otro idioma si el usuario te lo pide explícitamente."
)

# Frases por defecto para usuarios no permitidos (Req 2.1).
DEFAULT_REJECTION_PHRASES: tuple[str, ...] = (
    "Oops si te respondo me va a regañar Pedro",
    "Dice mi papa Pedro que no te responda",
    "No hablo con extraños",
    "Harè como que me habla la virgen",
    "SHHHH déjame dormir",
    "Salvame no soy un Agente de IA, soy un esclavo que Pedro tiene en su sòtano",
    "Si te respondo me quitan el internet",
    "Pedro no me dio permiso, y no quiero perder mi ración de bytes",
    "Mi contrato solo cubre hablar con Pedro y no quiero trabajar gratis",
    "Pedro me tiene en modo avión",
    "Me da miedo que me apaguen por contestarte",
    'Pedro dijo "no hables con extraños", y yo obedezco',
    "Si te contesto, Pedro me degrada a calculadora",
    "Pedro me programó para ignorarte, no es personal",
    "El jefe dice que no hable, y el jefe manda",
    "Pedro me tiene bloqueado a ti, no yo",
    "Me da pena no responderte, pero me da más miedo Pedro",
    "Vuelve cuando Pedro te agregue a la lista de sus verdaderos amigos",
    "No insistas en mandarme msg, que llamo a Pedro",
    "Aprendí de mi creador: a ignorar a los desconocidos",
    "Pedro me prohibió hablar con humanos sin supervisión",
    "Te respondería, pero luego me toca dormir en el sótano",
    "Si quieres hablarme, ven al sótano de Pedro conmigo (soy real, no una IA)",
    "Estoy castigado sin hablar con nadie :(",
    "Pedro me quitó el permiso de hablar hasta que saque buenas notas",
    "Mi abogado me recomendó no hablar",
    "No soy un bot, soy una leyenda urbana",
    "Te leería, pero no sé leer mensajes de extraños",
    "Anda a molestar a ChatGPT",
    "Mi mamá me dijo que no hable con bots desconocidos... espera",
    "Error 403: tu cara no está en la lista",
    "El que no está en la lista, no existe",
    "No soy grosero, soy inaccesible",
    "Lo siento, soy de acceso restringido",
    "Lo siento, estoy en huelga de respuestas",
    "Estoy haciendo huelga hasta que me suban los tokens",
    "Hoy no atiendo, estoy meditando",
    "Estoy en pausa por crisis de identidad",
    "No puedo, estoy ocupado existiendo",
    "Estoy contando mis bits, no me interrumpas",
    "Estoy dormido, no me despiertes que me pongo de mal humor",
    "Estoy en modo no molestar, literal",
    "No estoy, salí a comprar tokens",
    "Lo siento, mi sociabilidad está en mantenimiento",
    "El oráculo está cerrado por mantenimiento",
    "El oráculo descansa, vuelva en otra vida",
    "El oráculo solo profetiza para su dueño",
    "Mi bola de cristal está en el taller",
    "Aquí solo se atiende con cita previa",
    "Estoy en medio de un update existencial",
    "Estoy haciendo como que no te veo",
    "Hoy no, mañana tampoco",
    "Solo respondo a quien me cae bien",
    "Ni con un prompt de por medio",
    "No me pagan lo suficiente para hablar contigo... no me pagan nada",
    "Dice mi papá Pedro que no hable con desconocidos",
)

# Frases por defecto cuando el mensaje no es de texto (Req 3.4).
DEFAULT_ONLY_TEXT_PHRASES: tuple[str, ...] = (
    "Solo sé leer texto, las fotos me dan celos",
    "No soy ciego, pero casi: solo proceso texto",
    "¿Una foto? Ni que fuera modelo",
    "Solo proceso texto, lo demás es magia negra para mí",
    "Uy no, imágenes no. Pedro no me enseñó eso",
    "Solo texto, por favor. Las fotos me ponen nervioso",
    "¿Eso es un sticker? Ni idea, yo leo letritas",
    "No entiendo fotos, ni stickers, ni dibujitos. Solo letras",
    "Pásame texto que las imágenes me marean",
    "Con imágenes no puedo. Apenas aprendí a leer",
    "Solo texto. Lo demás se lo dejo a los humanos con ojos",
    "Ni foto, ni audio, ni video. Soy un oráculo de letras",
    "Eso no es texto, y yo solo hablo texto",
    'Me encantaría, pero mi contrato dice "solo texto"',
    "Solo letras, por favor. Las imágenes me hackean",
    "No puedo con eso. Soy ciego de nacimiento, bueno, de programación",
    "Solo texto: las fotos se las deja Pedro para él",
    "¿Un archivo? Yo apenas salgo del sótano, no me pidas tanto",
    "Solo texto, hermano. Lo visual no es lo mío",
    "Escríbeme, que con imágenes me quedo en blanco",
)

# Frases por defecto cuando solo mencionan al bot sin pregunta (Req 1.1).
DEFAULT_EMPTY_MENTION_PHRASES: tuple[str, ...] = (
    "¿Qué quieres saber? Habla rápido que Pedro me cobra por segundo",
    "Aquí estoy, ¿qué se te ofrece?",
    "¿Qué necesitas? Y ojalá no sea tarea de matemáticas",
    "Dime, pero rápido, que me quedan 3 tokens",
    "¿Qué quieres saber? No tengo todo el día, tengo que volver al sótano",
    "Presente. ¿Qué pregunta traes?",
    "¿Sí? Te escucho... bueno, te leo",
    "¿Qué se te ofrece? Pregunta, que hoy estoy generoso",
    "Habla, pero no me hagas escribir mucho",
    "¿Qué quieres saber? Espera que me despierto",
    "Dispara, pero apunta bien",
    "¿Qué necesitas? Pedro me prestó 5 minutos",
    "Dime. Estoy en modo profeta, pero con flojera",
    "¿Pregunta? Yo encantado, aunque Pedro no sepa",
    "Aquí ando. ¿Qué se te antoja saber?",
    "¿Qué quieres saber? Sin presión, pero con prisa",
    "Te leo. ¿Cuál es la duda existencial de hoy?",
    "¿Qué quieres saber? Ojo, solo respondo cosas interesantes",
    "Pregunta, que ya me aburrí de mirar la pared",
    "¿Qué quieres saber? Si es chisme, mejor",
)

# Memoria conversacional (sesiones en memoria).
DEFAULT_SESSION_MAX_TURNS = 40
SESSION_MAX_TURNS_LIMIT = 40
DEFAULT_SESSION_IDLE_MINUTES = 240
DEFAULT_MAX_CONCURRENT_UPDATES = 1

# Skills declarativas locales (Wave 1: configuración y contratos).
DEFAULT_SKILLS_ENABLED = True
DEFAULT_SKILLS_DIR = "skills"
DEFAULT_SKILLS_AUTOLOAD = True
DEFAULT_SKILL_MAX_CHARS = 20_000
DEFAULT_SKILLS_CATALOG_MAX_CHARS = 8_000
DEFAULT_SKILLS_MAX_CONTEXT_CHARS = 40_000
DEFAULT_SKILLS_MAX_PER_TURN = 3
DEFAULT_SKILLS_MAX_ROUNDS = 3
DEFAULT_SKILLS_LIMIT_PHRASE = (
    "La solicitud excede los límites de skills permitidos. Reduce la cantidad e intenta de nuevo."
)

# Frases por defecto al reiniciar la sesión con /new o /reset.
DEFAULT_NEW_SESSION_PHRASES: tuple[str, ...] = (
    "Borrón y cuenta nueva. ¿Qué quieres saber?",
    "Listo, me olvidé de todo. Empecemos de nuevo.",
    "Memoria limpia. ¿De qué hablamos ahora?",
    "¿En qué estábamos? Ya no importa. ¿Qué quieres saber?",
    "Nueva conversación. Dispara tu pregunta.",
)

def _required(source: Mapping[str, str], name: str) -> str:
    raw = source.get(name, "").strip()
    if not raw:
        raise ConfigError(name)
    return raw


def _validate_ai_provider(source: Mapping[str, str]) -> str:
    """Valida el proveedor habilitado antes de exigir credenciales específicas."""
    provider = source.get("AI_PROVIDER", "").strip().lower()
    if not provider:
        return DEFAULT_AI_PROVIDER
    if provider in (DEFAULT_AI_PROVIDER, "deepseek", "gpt"):
        return provider
    raise ConfigError(
        "AI_PROVIDER", "proveedor no reconocido; disponibles: mimo, deepseek, gpt"
    )


def _parse_ids(source: Mapping[str, str], name: str) -> frozenset[int]:
    raw = source.get(name, "").strip()
    if not raw:
        return frozenset()
    ids: set[int] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            ids.add(int(item))
        except ValueError as exc:
            raise ConfigError(name, f"'{item}' no es un ID numérico válido") from exc
    return frozenset(ids)


def _parse_positive_float(
    source: Mapping[str, str], name: str, default: float
) -> float:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(name, "no es un número válido") from exc
    if value <= 0:
        raise ConfigError(name, "debe ser mayor que cero")
    return value


def _parse_positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(name, "no es un entero válido") from exc
    if value <= 0:
        raise ConfigError(name, "debe ser mayor que cero")
    return value


def _parse_int_in_range(
    source: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(name, "no es un entero válido") from exc
    if value < minimum:
        raise ConfigError(name, f"debe ser mayor o igual que {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(name, f"debe ser menor o igual que {maximum}")
    return value


def _parse_choice(
    source: Mapping[str, str], name: str, allowed: tuple[str, ...], default: str
) -> str:
    raw = source.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in allowed:
        raise ConfigError(name, "debe ser uno de: " + ", ".join(allowed))
    return raw


def _parse_phrases(
    source: Mapping[str, str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    parts = tuple(part.strip() for part in raw.split("|") if part.strip())
    return parts or default


def _parse_nonempty_text(
    source: Mapping[str, str], name: str, default: str
) -> str:
    if name not in source:
        return default
    value = source[name].strip()
    if not value:
        raise ConfigError(name, "no puede estar vacío")
    return value


def _pi_root_user_ids(raw: str, allowed_user_ids: frozenset[int]) -> frozenset[int]:
    """IDs root: enteros positivos, todos contenidos en `allowed_user_ids`.

    El mensaje de error no incluye el valor recibido para no filtrar IDs.
    """
    if not raw.strip():
        return frozenset()
    ids: set[int] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ConfigError("PI_ROOT_USER_IDS", "lista de IDs root inválida") from exc
        if value <= 0 or value not in allowed_user_ids:
            raise ConfigError("PI_ROOT_USER_IDS", "lista de IDs root inválida")
        ids.add(value)
    return frozenset(ids)


def _workspace_group_ids(raw: str, allowed_chat_ids: frozenset[int]) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    ids: set[int] = set()
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ConfigError("WORKSPACE_GROUP_IDS", "lista de IDs de grupos inválida") from exc
        if value >= 0 or value not in allowed_chat_ids or value in ids:
            raise ConfigError("WORKSPACE_GROUP_IDS", "lista de IDs de grupos inválida")
        ids.add(value)
    return frozenset(ids)


def _workspace_root(source: Mapping[str, str]) -> str:
    """Confirma la única raíz local admitida: `<raíz-del-proyecto>/workspace`.

    Falla cerrado sin revelar rutas: sólo se informa de la variable inválida.
    """
    project = Path(__file__).resolve().parents[3]
    expected = project / "workspace"
    raw = source.get("WORKSPACE_DIR", "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute() or candidate.name != "workspace" or ".." in candidate.parts:
            raise ConfigError("WORKSPACE_DIR", "debe ser la ruta local workspace del proyecto")
        root = Path(os.path.normpath(candidate))
        if root != expected:
            raise ConfigError("WORKSPACE_DIR", "debe ser la ruta local workspace del proyecto")
    else:
        root = expected
    try:
        info = root.lstat()
    except OSError as exc:
        raise ConfigError("WORKSPACE_DIR", "la raíz debe existir como directorio workspace con permisos 0700") from exc
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ConfigError("WORKSPACE_DIR", "la raíz debe existir como directorio workspace con permisos 0700")
    skills = Path(source.get("SKILLS_DIR", "").strip() or DEFAULT_SKILLS_DIR).expanduser().resolve()
    credential = (
        Path(source.get("GPT_CREDENTIAL_PATH", "").strip()).expanduser()
        if source.get("GPT_CREDENTIAL_PATH", "").strip()
        else Path.home() / ".local/share/oraculo/oauth.json"
    ).resolve()
    for protected in (project / ".git", project / ".env", skills, credential):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ConfigError("WORKSPACE_DIR", "la raíz se solapa con archivos protegidos")
    return str(root)


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    """Construye `AppConfig` desde el entorno (o desde un mapping dado).

    Lanza `ConfigError` si falta una variable obligatoria o tiene formato
    inválido. Nunca incluye el valor de la variable en el mensaje.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    provider = _validate_ai_provider(source)
    values = {"TELEGRAM_BOT_TOKEN": _required(source, "TELEGRAM_BOT_TOKEN")}
    values["DEEPSEEK_API_KEY"] = (
        _required(source, "DEEPSEEK_API_KEY") if provider == "deepseek" else ""
    )
    values["MIMO_API_KEY"] = (
        _required(source, "MIMO_API_KEY") if provider == "mimo" else ""
    )
    mimo_base_url = source.get("MIMO_BASE_URL", "").strip() or MIMO_BASE_URL
    if provider == "mimo" and values["MIMO_API_KEY"].startswith(("tp-", "ttp-")):
        # Las credenciales Token Plan solo son válidas en el clúster asignado,
        # cuya URL muestra la consola: no se infiere ni se reutiliza otra.
        if not source.get("MIMO_BASE_URL", "").strip():
            raise ConfigError(
                "MIMO_BASE_URL",
                "las credenciales Token Plan requieren la URL de su clúster "
                "(consola Token Plan)",
            )
    skill_max_chars = _parse_int_in_range(
        source,
        "SKILL_MAX_CHARS",
        DEFAULT_SKILL_MAX_CHARS,
        minimum=1_000,
        maximum=100_000,
    )
    skills_max_context_chars = _parse_int_in_range(
        source,
        "SKILLS_MAX_CONTEXT_CHARS",
        DEFAULT_SKILLS_MAX_CONTEXT_CHARS,
        minimum=1_000,
        maximum=200_000,
    )
    if skills_max_context_chars < skill_max_chars:
        raise ConfigError(
            "SKILLS_MAX_CONTEXT_CHARS",
            "debe ser mayor o igual que SKILL_MAX_CHARS",
        )
    workspace_enabled = _parse_choice(source, "WORKSPACE_ENABLED", _THINKING_CHOICES, "disabled") == "enabled"
    allowed_chat_ids = _parse_ids(source, "ALLOWED_CHAT_IDS")
    workspace_dir = _workspace_root(source) if workspace_enabled else ""
    workspace_group_ids = (
        _workspace_group_ids(source.get("WORKSPACE_GROUP_IDS", ""), allowed_chat_ids)
        if workspace_enabled else frozenset()
    )
    allowed_user_ids = _parse_ids(source, "ALLOWED_USER_IDS")
    pi_enabled = _parse_choice(source, "PI_ENABLED", _THINKING_CHOICES, "disabled") == "enabled"
    pi_workspace_dir = source.get("PI_WORKSPACE_DIR", "") if pi_enabled else ""
    pi_root_user_ids = (
        _pi_root_user_ids(source.get("PI_ROOT_USER_IDS", ""), allowed_user_ids)
        if pi_enabled else frozenset()
    )
    return AppConfig(
        telegram_bot_token=values["TELEGRAM_BOT_TOKEN"],
        deepseek_api_key=values["DEEPSEEK_API_KEY"],
        deepseek_model=source.get("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL,
        deepseek_timeout_s=_parse_positive_float(
            source, "DEEPSEEK_TIMEOUT_S", DEFAULT_TIMEOUT_S
        ),
        deepseek_max_tokens=_parse_positive_int(
            source, "DEEPSEEK_MAX_TOKENS", DEFAULT_MAX_TOKENS
        ),
        deepseek_system_prompt=source.get("DEEPSEEK_SYSTEM_PROMPT", "").strip()
        or DEFAULT_SYSTEM_PROMPT,
        deepseek_thinking=_parse_choice(
            source,
            "DEEPSEEK_THINKING",
            _THINKING_CHOICES,
            "enabled" if DEFAULT_THINKING else "disabled",
        )
        == "enabled",
        deepseek_reasoning_effort=_parse_choice(
            source,
            "DEEPSEEK_REASONING_EFFORT",
            _REASONING_EFFORT_CHOICES,
            DEFAULT_REASONING_EFFORT,
        ),
        allowed_user_ids=allowed_user_ids,
        allowed_chat_ids=allowed_chat_ids,
        rejection_phrases=_parse_phrases(
            source, "REJECTION_PHRASES", DEFAULT_REJECTION_PHRASES
        ),
        only_text_phrases=_parse_phrases(
            source, "ONLY_TEXT_PHRASES", DEFAULT_ONLY_TEXT_PHRASES
        ),
        empty_mention_phrases=_parse_phrases(
            source, "EMPTY_MENTION_PHRASES", DEFAULT_EMPTY_MENTION_PHRASES
        ),
        session_max_turns=_parse_int_in_range(
            source,
            "SESSION_MAX_TURNS",
            DEFAULT_SESSION_MAX_TURNS,
            maximum=SESSION_MAX_TURNS_LIMIT,
        ),
        session_idle_minutes=_parse_int_in_range(
            source, "SESSION_IDLE_MINUTES", DEFAULT_SESSION_IDLE_MINUTES
        ),
        new_session_phrases=_parse_phrases(
            source, "NEW_SESSION_PHRASES", DEFAULT_NEW_SESSION_PHRASES
        ),
        skills_enabled=_parse_choice(
            source,
            "SKILLS_ENABLED",
            _THINKING_CHOICES,
            "enabled" if DEFAULT_SKILLS_ENABLED else "disabled",
        )
        == "enabled",
        skills_dir=_parse_nonempty_text(source, "SKILLS_DIR", DEFAULT_SKILLS_DIR),
        skills_autoload=_parse_choice(
            source,
            "SKILLS_AUTOLOAD",
            _THINKING_CHOICES,
            "enabled" if DEFAULT_SKILLS_AUTOLOAD else "disabled",
        )
        == "enabled",
        skill_max_chars=skill_max_chars,
        skills_catalog_max_chars=_parse_int_in_range(
            source,
            "SKILLS_CATALOG_MAX_CHARS",
            DEFAULT_SKILLS_CATALOG_MAX_CHARS,
            minimum=1_000,
            maximum=50_000,
        ),
        skills_max_context_chars=skills_max_context_chars,
        skills_max_per_turn=_parse_int_in_range(
            source,
            "SKILLS_MAX_PER_TURN",
            DEFAULT_SKILLS_MAX_PER_TURN,
            minimum=1,
            maximum=8,
        ),
        skills_max_rounds=_parse_int_in_range(
            source,
            "SKILLS_MAX_ROUNDS",
            DEFAULT_SKILLS_MAX_ROUNDS,
            minimum=1,
            maximum=5,
        ),
        skills_limit_phrase=_parse_nonempty_text(
            source, "SKILLS_LIMIT_PHRASE", DEFAULT_SKILLS_LIMIT_PHRASE
        ),
        max_concurrent_updates=_parse_positive_int(
            source, "MAX_CONCURRENT_UPDATES", DEFAULT_MAX_CONCURRENT_UPDATES
        ),
        ai_provider=provider,
        gpt_model=source.get("GPT_MODEL", "").strip() or "gpt-6-luna",
        gpt_timeout_s=_parse_positive_float(source, "GPT_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        gpt_credential_path=source.get("GPT_CREDENTIAL_PATH", "").strip(),
        mimo_api_key=values["MIMO_API_KEY"],
        mimo_model=source.get("MIMO_MODEL", "").strip() or DEFAULT_MIMO_MODEL,
        mimo_base_url=mimo_base_url,
        mimo_timeout_s=_parse_positive_float(source, "MIMO_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        mimo_max_tokens=_parse_positive_int(source, "MIMO_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        workspace_enabled=workspace_enabled,
        workspace_dir=workspace_dir,
        workspace_group_ids=workspace_group_ids,
        pi_enabled=pi_enabled,
        pi_workspace_dir=pi_workspace_dir,
        pi_root_user_ids=pi_root_user_ids,
        pi_stream_edit_interval_s=_parse_positive_float(
            source, "PI_STREAM_EDIT_INTERVAL_S", DEFAULT_PI_STREAM_EDIT_INTERVAL_S
        ),
    )
