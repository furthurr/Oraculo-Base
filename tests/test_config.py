"""Tests de carga y validación de configuración (Req 4.1, 4.2, 4.4)."""

from __future__ import annotations

import pytest
from pathlib import Path

from oraculo.domain.errors import ConfigError
from oraculo.infrastructure.config import (
    DEFAULT_EMPTY_MENTION_PHRASES,
    DEFAULT_MAX_CONCURRENT_UPDATES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NEW_SESSION_PHRASES,
    DEFAULT_ONLY_TEXT_PHRASES,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REJECTION_PHRASES,
    DEFAULT_SESSION_IDLE_MINUTES,
    DEFAULT_SESSION_MAX_TURNS,
    DEFAULT_SKILL_MAX_CHARS,
    DEFAULT_SKILLS_AUTOLOAD,
    DEFAULT_SKILLS_CATALOG_MAX_CHARS,
    DEFAULT_SKILLS_DIR,
    DEFAULT_SKILLS_ENABLED,
    DEFAULT_SKILLS_LIMIT_PHRASE,
    DEFAULT_SKILLS_MAX_CONTEXT_CHARS,
    DEFAULT_SKILLS_MAX_PER_TURN,
    DEFAULT_SKILLS_MAX_ROUNDS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT_S,
    load_config,
)

BASE_ENV = {"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-test"}


def test_config_minima_aplica_defaults():
    config = load_config(BASE_ENV)

    assert config.telegram_bot_token == "123:abc"
    assert config.deepseek_api_key == "sk-test"
    assert config.deepseek_model == DEFAULT_MODEL
    assert config.deepseek_timeout_s == DEFAULT_TIMEOUT_S
    assert config.deepseek_max_tokens == DEFAULT_MAX_TOKENS
    assert config.deepseek_system_prompt == DEFAULT_SYSTEM_PROMPT
    assert config.deepseek_thinking is DEFAULT_THINKING
    assert config.deepseek_reasoning_effort == DEFAULT_REASONING_EFFORT
    assert config.allowed_user_ids == frozenset()
    assert config.allowed_chat_ids == frozenset()
    assert config.rejection_phrases == DEFAULT_REJECTION_PHRASES
    assert config.only_text_phrases == DEFAULT_ONLY_TEXT_PHRASES
    assert config.empty_mention_phrases == DEFAULT_EMPTY_MENTION_PHRASES
    assert not hasattr(config, "ai_error_phrases")
    assert config.session_max_turns == DEFAULT_SESSION_MAX_TURNS
    assert config.session_idle_minutes == DEFAULT_SESSION_IDLE_MINUTES
    assert config.new_session_phrases == DEFAULT_NEW_SESSION_PHRASES
    assert config.max_concurrent_updates == DEFAULT_MAX_CONCURRENT_UPDATES
    assert config.skills_enabled is DEFAULT_SKILLS_ENABLED
    assert config.skills_dir == DEFAULT_SKILLS_DIR
    assert config.skills_autoload is DEFAULT_SKILLS_AUTOLOAD
    assert config.skill_max_chars == DEFAULT_SKILL_MAX_CHARS
    assert config.skills_catalog_max_chars == DEFAULT_SKILLS_CATALOG_MAX_CHARS
    assert config.skills_max_context_chars == DEFAULT_SKILLS_MAX_CONTEXT_CHARS
    assert config.skills_max_per_turn == DEFAULT_SKILLS_MAX_PER_TURN
    assert config.skills_max_rounds == DEFAULT_SKILLS_MAX_ROUNDS
    assert config.skills_limit_phrase == DEFAULT_SKILLS_LIMIT_PHRASE


def test_max_concurrent_updates_acepta_un_limite_positivo() -> None:
    config = load_config({**BASE_ENV, "MAX_CONCURRENT_UPDATES": "4"})

    assert config.max_concurrent_updates == 4


@pytest.mark.parametrize("value", ["0", "-1", "no-es-numero"])
def test_max_concurrent_updates_invalido_falla_sin_exponer_el_valor(
    value: str,
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "MAX_CONCURRENT_UPDATES": value})

    assert excinfo.value.variable == "MAX_CONCURRENT_UPDATES"
    assert value not in str(excinfo.value)


def test_config_transcripcion_local_ignora_clave_y_timeout_legacy():
    config = load_config({
        **BASE_ENV,
        "DEEPGRAM_API_KEY": "legacy-test-value",
        "AUDIO_TRANSCRIPTION_TIMEOUT_S": "not-a-timeout",
    })

    assert config.ai_provider == "deepseek"
    assert not hasattr(config, "deepgram_api_key")
    assert not hasattr(config, "audio_transcription_timeout_s")


def test_ai_provider_deepseek_explicito_conserva_el_proveedor_actual() -> None:
    config = load_config({**BASE_ENV, "AI_PROVIDER": "deepseek"})

    assert config.deepseek_api_key == "sk-test"


@pytest.mark.parametrize("provider", ["future-llm"])
def test_ai_provider_no_habilitado_lanza_config_error_sin_exponer_valor(
    provider: str,
) -> None:
    env = {**BASE_ENV, "AI_PROVIDER": provider}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == "AI_PROVIDER"
    assert provider not in str(excinfo.value)
    assert BASE_ENV["DEEPSEEK_API_KEY"] not in str(excinfo.value)
    assert BASE_ENV["TELEGRAM_BOT_TOKEN"] not in str(excinfo.value)


def test_ai_provider_gpt_no_exige_api_key_de_deepseek() -> None:
    env = {"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "gpt"}

    config = load_config(env)

    assert config.ai_provider == "gpt"
    assert config.deepseek_api_key == ""
    assert config.gpt_model == "gpt-6-luna"


def test_ai_provider_gpt_permite_modelo_explicito() -> None:
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "gpt",
        "GPT_MODEL": "gpt-5.4-mini",
    })
    assert config.gpt_model == "gpt-5.4-mini"


def test_ai_provider_default_es_mimo() -> None:
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "MIMO_API_KEY": "mk-test"})
    assert config.ai_provider == "mimo"


def test_ai_provider_mimo_aplica_defaults_de_modelo() -> None:
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "MIMO_API_KEY": "mk-test"})

    assert config.mimo_api_key == "mk-test"
    assert config.mimo_model == "xiaomi/mimo-v2.6-flash"
    assert config.mimo_base_url == "https://api.xiaomimimo.com/v1"
    assert config.mimo_timeout_s == DEFAULT_TIMEOUT_S
    assert config.mimo_max_tokens == DEFAULT_MAX_TOKENS
    assert config.deepseek_api_key == ""


def test_ai_provider_mimo_permite_modelo_y_endpoint_explicitos() -> None:
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo",
        "MIMO_API_KEY": "mk-test", "MIMO_MODEL": "xiaomi/mimo-v2.6-pro",
        "MIMO_BASE_URL": "https://proxy.example/v1", "MIMO_TIMEOUT_S": "15",
        "MIMO_MAX_TOKENS": "2048",
    })

    assert config.mimo_model == "xiaomi/mimo-v2.6-pro"
    assert config.mimo_base_url == "https://proxy.example/v1"
    assert config.mimo_timeout_s == 15.0
    assert config.mimo_max_tokens == 2048


def test_ai_provider_mimo_exige_mimo_api_key() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo"})

    assert excinfo.value.variable == "MIMO_API_KEY"


def test_ai_provider_mimo_no_exige_api_key_de_deepseek() -> None:
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo",
                          "MIMO_API_KEY": "mk-test"})

    assert config.ai_provider == "mimo"
    assert config.deepseek_api_key == ""


def test_mimo_token_plan_exige_base_url_de_cluster() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo",
                     "MIMO_API_KEY": "tp-ejemplo"})

    assert excinfo.value.variable == "MIMO_BASE_URL"
    assert "tp-ejemplo" not in str(excinfo.value)


def test_mimo_token_plan_con_base_url_de_cluster_se_acepta() -> None:
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo",
                          "MIMO_API_KEY": "ttp-ejemplo",
                          "MIMO_BASE_URL": "https://token-plan-sgp.xiaomimimo.com/v1"})

    assert config.mimo_base_url == "https://token-plan-sgp.xiaomimimo.com/v1"


def test_mimo_pago_por_uso_no_exige_base_url() -> None:
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo",
                          "MIMO_API_KEY": "sk-ejemplo"})

    assert config.mimo_base_url == "https://api.xiaomimimo.com/v1"


def test_config_parsea_listas_de_ids():
    env = {**BASE_ENV, "ALLOWED_USER_IDS": " 1, 2 ,3 ", "ALLOWED_CHAT_IDS": "-1001,-1002"}

    config = load_config(env)

    assert config.allowed_user_ids == frozenset({1, 2, 3})
    assert config.allowed_chat_ids == frozenset({-1001, -1002})


def test_config_parsea_frases_personalizadas():
    env = {**BASE_ENV, "REJECTION_PHRASES": "uno | dos | tres"}

    config = load_config(env)

    assert config.rejection_phrases == ("uno", "dos", "tres")


def test_system_prompt_personalizado_se_recorta():
    env = {**BASE_ENV, "DEEPSEEK_SYSTEM_PROMPT": "  Responde en inglés  "}

    config = load_config(env)

    assert config.deepseek_system_prompt == "Responde en inglés"


@pytest.mark.parametrize("valor", ["", "   "])
def test_system_prompt_vacio_usa_default(valor):
    env = {**BASE_ENV, "DEEPSEEK_SYSTEM_PROMPT": valor}

    config = load_config(env)

    assert config.deepseek_system_prompt == DEFAULT_SYSTEM_PROMPT


def test_config_parsea_frases_de_mensajes_automaticos():
    env = {
        **BASE_ENV,
        "ONLY_TEXT_PHRASES": "a | b",
        "EMPTY_MENTION_PHRASES": "c | d",
        "AI_ERROR_PHRASES": "e | f",
    }

    config = load_config(env)

    assert config.only_text_phrases == ("a", "b")
    assert config.empty_mention_phrases == ("c", "d")
    assert not hasattr(config, "ai_error_phrases")


@pytest.mark.parametrize(
    "valor,esperado",
    [("enabled", True), ("disabled", False), ("ENABLED", True), (" Disabled ", False)],
)
def test_config_parsea_thinking(valor, esperado):
    env = {**BASE_ENV, "DEEPSEEK_THINKING": valor}

    config = load_config(env)

    assert config.deepseek_thinking is esperado


@pytest.mark.parametrize("valor", ["low", "high", "max", "MAX"])
def test_config_parsea_effort(valor):
    env = {**BASE_ENV, "DEEPSEEK_REASONING_EFFORT": valor}

    config = load_config(env)

    assert config.deepseek_reasoning_effort == valor.lower()


@pytest.mark.parametrize(
    "variable,valor",
    [("DEEPSEEK_THINKING", "si"), ("DEEPSEEK_REASONING_EFFORT", "ultra")],
)
def test_opcion_invalida_lanza_config_error(variable, valor):
    env = {**BASE_ENV, variable: valor}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == variable


@pytest.mark.parametrize("variable", ["TELEGRAM_BOT_TOKEN", "DEEPSEEK_API_KEY"])
def test_variable_obligatoria_ausente_lanza_config_error(variable):
    env = {name: value for name, value in BASE_ENV.items() if name != variable}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == variable


def test_error_de_config_no_expone_valores_de_secretos():
    env = {**BASE_ENV, "DEEPSEEK_TIMEOUT_S": "no-es-numero"}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == "DEEPSEEK_TIMEOUT_S"
    assert BASE_ENV["TELEGRAM_BOT_TOKEN"] not in str(excinfo.value)
    assert BASE_ENV["DEEPSEEK_API_KEY"] not in str(excinfo.value)


def test_id_no_numerico_lanza_config_error():
    env = {**BASE_ENV, "ALLOWED_USER_IDS": "1, abc"}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == "ALLOWED_USER_IDS"


@pytest.mark.parametrize(
    "variable,valor",
    [("DEEPSEEK_TIMEOUT_S", "0"), ("DEEPSEEK_MAX_TOKENS", "-5")],
)
def test_valor_no_positivo_lanza_config_error(variable, valor):
    env = {**BASE_ENV, variable: valor}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == variable


# --- Memoria conversacional (Req 6.1, 6.2, 6.4) ---


def test_config_parsea_variables_de_sesion():
    env = {**BASE_ENV, "SESSION_MAX_TURNS": "5", "SESSION_IDLE_MINUTES": "60"}

    config = load_config(env)

    assert config.session_max_turns == 5
    assert config.session_idle_minutes == 60


def test_session_max_turns_cero_es_valido():
    config = load_config({**BASE_ENV, "SESSION_MAX_TURNS": "0"})

    assert config.session_max_turns == 0


def test_cuarenta_turnos_es_valido_y_predeterminado():
    assert load_config(BASE_ENV).session_max_turns == 40
    assert load_config({**BASE_ENV, "SESSION_MAX_TURNS": "40"}).session_max_turns == 40


def test_session_idle_cero_es_valido():
    config = load_config({**BASE_ENV, "SESSION_IDLE_MINUTES": "0"})

    assert config.session_idle_minutes == 0


@pytest.mark.parametrize(
    "variable,valor",
    [
        ("SESSION_MAX_TURNS", "41"),
        ("SESSION_MAX_TURNS", "-1"),
        ("SESSION_MAX_TURNS", "abc"),
        ("SESSION_IDLE_MINUTES", "-5"),
        ("SESSION_IDLE_MINUTES", "no"),
    ],
)
def test_sesion_invalida_lanza_config_error(variable, valor):
    env = {**BASE_ENV, variable: valor}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == variable


def test_config_parsea_frases_de_sesion_nueva():
    config = load_config({**BASE_ENV, "NEW_SESSION_PHRASES": "a | b"})

    assert config.new_session_phrases == ("a", "b")


# --- Skills (Req 6.1, 6.2) ---


def test_config_parsea_variables_de_skills():
    config = load_config(
        {
            **BASE_ENV,
            "SKILLS_ENABLED": "disabled",
            "SKILLS_DIR": " custom-skills ",
            "SKILLS_AUTOLOAD": "disabled",
            "SKILL_MAX_CHARS": "1000",
            "SKILLS_CATALOG_MAX_CHARS": "1000",
            "SKILLS_MAX_CONTEXT_CHARS": "1000",
            "SKILLS_MAX_PER_TURN": "8",
            "SKILLS_MAX_ROUNDS": "5",
            "SKILLS_LIMIT_PHRASE": " Demasiadas skills. ",
        }
    )

    assert config.skills_enabled is False
    assert config.skills_dir == "custom-skills"
    assert config.skills_autoload is False
    assert config.skill_max_chars == 1000
    assert config.skills_catalog_max_chars == 1000
    assert config.skills_max_context_chars == 1000
    assert config.skills_max_per_turn == 8
    assert config.skills_max_rounds == 5
    assert config.skills_limit_phrase == "Demasiadas skills."


@pytest.mark.parametrize(
    "variable,valor",
    [
        ("SKILLS_ENABLED", "yes"),
        ("SKILLS_AUTOLOAD", "yes"),
        ("SKILLS_DIR", "   "),
        ("SKILL_MAX_CHARS", "999"),
        ("SKILL_MAX_CHARS", "100001"),
        ("SKILLS_CATALOG_MAX_CHARS", "999"),
        ("SKILLS_CATALOG_MAX_CHARS", "50001"),
        ("SKILLS_MAX_CONTEXT_CHARS", "200001"),
        ("SKILLS_MAX_PER_TURN", "0"),
        ("SKILLS_MAX_PER_TURN", "9"),
        ("SKILLS_MAX_ROUNDS", "0"),
        ("SKILLS_MAX_ROUNDS", "6"),
        ("SKILLS_LIMIT_PHRASE", "   "),
    ],
)
def test_variable_de_skills_invalida_lanza_config_error(variable: str, valor: str):
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, variable: valor})

    assert excinfo.value.variable == variable
    if valor.strip():
        assert valor.strip() not in str(excinfo.value)


def test_contexto_de_skills_no_puede_ser_menor_que_una_skill():
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            {
                **BASE_ENV,
                "SKILL_MAX_CHARS": "2000",
                "SKILLS_MAX_CONTEXT_CHARS": "1999",
            }
        )

    assert excinfo.value.variable == "SKILLS_MAX_CONTEXT_CHARS"


def test_workspace_apagado_no_necesita_directorio() -> None:
    config = load_config({**BASE_ENV, "WORKSPACE_DIR": "/no/existe/workspace"})
    assert config.workspace_enabled is False
    assert config.workspace_dir == ""
    assert config.workspace_group_ids == frozenset()


def test_workspace_habilitado_usa_grupos_habilitados(tmp_project: Path) -> None:
    root = tmp_project / "workspace"
    root.mkdir(mode=0o700)
    config = load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled",
                          "WORKSPACE_DIR": str(root), "ALLOWED_CHAT_IDS": "-100,-200",
                          "WORKSPACE_GROUP_IDS": "-100,-200"})
    assert config.workspace_enabled is True
    assert config.workspace_dir == str(root)
    assert config.workspace_group_ids == frozenset({-100, -200})


@pytest.mark.parametrize("extra,variable", [
    ({"WORKSPACE_DIR": "relative/workspace"}, "WORKSPACE_DIR"),
    ({"WORKSPACE_DIR": "/no/existe/workspace"}, "WORKSPACE_DIR"),
    ({"WORKSPACE_DIR": "/tmp/not-workspace"}, "WORKSPACE_DIR"),
    ({"WORKSPACE_GROUP_IDS": "100"}, "WORKSPACE_GROUP_IDS"),
    ({"WORKSPACE_GROUP_IDS": "-300"}, "WORKSPACE_GROUP_IDS"),
    ({"WORKSPACE_GROUP_IDS": "-100,-100"}, "WORKSPACE_GROUP_IDS"),
    ({"WORKSPACE_ENABLED": "yes"}, "WORKSPACE_ENABLED"),
])
def test_workspace_rechaza_configuracion_invalida(tmp_project: Path, extra: dict[str, str], variable: str) -> None:
    root = tmp_project / "workspace"
    root.mkdir(mode=0o700)
    env = {**BASE_ENV, "WORKSPACE_ENABLED": "enabled", "WORKSPACE_DIR": str(root),
           "ALLOWED_CHAT_IDS": "-100", **extra}
    with pytest.raises(ConfigError) as excinfo:
        load_config(env)
    assert excinfo.value.variable == variable
    assert str(root) not in str(excinfo.value)


def test_workspace_local_se_acepta_y_rechaza_solapes(tmp_project: Path) -> None:
    root = tmp_project / "workspace"
    root.mkdir(mode=0o700)
    config = load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled", "WORKSPACE_DIR": str(root)})
    assert config.workspace_dir == str(root)
    assert load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled"}).workspace_dir == str(root)
    skills = tmp_project / "skills"
    skills.mkdir()
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled",
                     "WORKSPACE_DIR": str(skills / "workspace"), "SKILLS_DIR": str(skills)})
    assert excinfo.value.variable == "WORKSPACE_DIR"
    assert str(skills / "workspace") not in str(excinfo.value)
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled",
                     "WORKSPACE_DIR": str(root), "SKILLS_DIR": str(root)})
    assert excinfo.value.variable == "WORKSPACE_DIR"
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled",
                     "WORKSPACE_DIR": str(root), "GPT_CREDENTIAL_PATH": str(root / "oauth.json")})
    assert excinfo.value.variable == "WORKSPACE_DIR"
    assert str(root) not in str(excinfo.value)


def test_workspace_exige_raiz_local_0700(tmp_project: Path) -> None:
    root = tmp_project / "workspace"
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled", "WORKSPACE_DIR": str(root)})
    assert excinfo.value.variable == "WORKSPACE_DIR"
    assert str(root) not in str(excinfo.value)
    root.mkdir(mode=0o700)
    root.chmod(0o755)
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled", "WORKSPACE_DIR": str(root)})
    assert excinfo.value.variable == "WORKSPACE_DIR"
    root.chmod(0o700)
    assert load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled",
                        "WORKSPACE_DIR": str(root)}).workspace_dir == str(root)


def test_workspace_rechaza_symlink(tmp_project: Path) -> None:
    target = tmp_project / "real"
    target.mkdir(mode=0o700)
    root = tmp_project / "workspace"
    root.symlink_to(target)
    with pytest.raises(ConfigError) as excinfo:
        load_config({**BASE_ENV, "WORKSPACE_ENABLED": "enabled", "WORKSPACE_DIR": str(root)})
    assert excinfo.value.variable == "WORKSPACE_DIR"
    assert str(root) not in str(excinfo.value)


PI_BASE_ENV = {
    **BASE_ENV,
    "PI_ENABLED": "enabled",
    "ALLOWED_USER_IDS": "11,22,33",
}


def test_pi_root_user_ids_vacio_es_frozenset_vacio():
    config = load_config(PI_BASE_ENV)

    assert config.pi_root_user_ids == frozenset()


def test_pi_root_user_ids_lista_valida():
    env = {**PI_BASE_ENV, "PI_ROOT_USER_IDS": " 11 , 22 "}

    config = load_config(env)

    assert config.pi_root_user_ids == frozenset({11, 22})


@pytest.mark.parametrize("valor", ["11,abc", "11,0", "11,-5"])
def test_pi_root_user_ids_valor_invalido_lanza_config_error(valor):
    env = {**PI_BASE_ENV, "PI_ROOT_USER_IDS": valor}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == "PI_ROOT_USER_IDS"
    # No filtra el valor recibido en el mensaje.
    assert valor not in str(excinfo.value)


def test_pi_root_user_id_fuera_de_allowed_lanza_config_error():
    env = {**PI_BASE_ENV, "PI_ROOT_USER_IDS": "11,99"}

    with pytest.raises(ConfigError) as excinfo:
        load_config(env)

    assert excinfo.value.variable == "PI_ROOT_USER_IDS"


def test_pi_root_user_ids_ignorado_si_pi_disabled():
    env = {
        **BASE_ENV,
        "ALLOWED_USER_IDS": "11",
        "PI_ENABLED": "disabled",
        "PI_ROOT_USER_IDS": "abc,0,-1",
    }

    config = load_config(env)

    assert config.pi_root_user_ids == frozenset()
