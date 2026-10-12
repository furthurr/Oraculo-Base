# Oraculo

Bot de Telegram en Python que conversa con proveedores de IA configurables. Usa
Xiaomi MiMo por defecto y también admite DeepSeek y una integración experimental
con ChatGPT/Codex.

## Funciones

- Acceso restringido por listas de usuarios y chats autorizados.
- Memoria conversacional temporal, con resumen automático y comandos `/new` y
  `/reset`.
- Procesamiento de adjuntos según el proveedor y transcripción local de audio.
- Skills locales en `skills/<nombre>/SKILL.md`; son instrucciones declarativas,
  no código ejecutable.
- Workspace documental opcional y modo agente Pi opcional.

Las conversaciones se mantienen en memoria y se pierden al reiniciar. El
workspace, cuando se activa, guarda documentos localmente y estos persisten entre
reinicios.

## Requisitos

- Python 3.12 o superior.
- Un token de bot de Telegram.
- Una credencial para el proveedor elegido.

## Instalación

Desde la raíz del proyecto:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Edita `.env` y configura `TELEGRAM_BOT_TOKEN`, `AI_PROVIDER`, la clave del
proveedor y las listas `ALLOWED_USER_IDS` y `ALLOWED_CHAT_IDS`. Las claves
disponibles son `MIMO_API_KEY`, `DEEPSEEK_API_KEY` y, para GPT, una autorización
local de Codex. No compartas ni subas `.env`.

Valores de `AI_PROVIDER`: `mimo` (predeterminado), `deepseek` o `gpt`. La
integración GPT/Codex usa un mecanismo experimental no oficial. Para autorizarla,
activa primero el entorno virtual y ejecuta `./login-gpt.sh` en el equipo donde
correrá el bot.

## Ejecución

```sh
./restart.sh
```

El script inicia el bot en segundo plano. Para consultar o detenerlo:

```sh
./restart.sh status
./restart.sh stop
```

## Pruebas

```sh
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Configuración opcional

- `WORKSPACE_ENABLED=enabled` habilita documentos locales. La carpeta `workspace/`
  debe crearse con permisos `0700` y no se incluye en el repositorio.
- `PI_ENABLED=enabled` habilita el modo agente Pi en chats privados. Limita
  `PI_ROOT_USER_IDS` a usuarios de confianza: el acceso a la raíz permite al
  agente operar sobre los archivos del proyecto.

El resto de opciones y sus valores de ejemplo están en `.env.example`.

## Autor

<a href="https://furthurr.github.io/" target="_blank" rel="noopener noreferrer">Pedro G. V. @furthurr</a>

- GitHub: https://github.com/furthurr
- Email: pedrogvas@gmail.com

## Licencia

MIT. Consulta el archivo `LICENSE`.
