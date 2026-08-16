"""Bale channel — Telegram-compatible adapter for Bale messenger (https://bale.ai).

Bale's Bot API is based on Telegram's Bot API with minor changes.
This channel inherits from TelegramChannel and points the HTTP client at
Bale's API servers (tapi.bale.ai) instead of Telegram's.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import Field
from telegram import (
    BotCommand,
    Update,
)
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from src.channels.bus.events import OutboundMessage
from src.channels.bus.queue import MessageBus
from src.channels.telegram import (
    TelegramChannel,
    TelegramConfig,
    _StreamBuf,
    _SEND_MAX_RETRIES,
    _SEND_RETRY_BASE_DELAY,
    _STREAM_EDIT_INTERVAL_DEFAULT,
    TELEGRAM_MAX_MESSAGE_LEN,
    TELEGRAM_HTML_MAX_LEN,
)

# ---------------------------------------------------------------------------
# Bale API endpoints
# ---------------------------------------------------------------------------
BALE_API_BASE_URL = "https://tapi.bale.ai/bot"
BALE_API_FILE_URL = "https://tapi.bale.ai/file/bot"


class BaleConfig(TelegramConfig):
    """Bale channel configuration — defaults tuned for Bale's API."""

    # Bale likely doesn't support Bot API 10.1 (sendRichMessage)
    rich_messages: bool = False


class BaleChannel(TelegramChannel):
    """Bale messenger channel adapter.

    Shares the same message handling, streaming, media, and inline-keyboard
    logic as TelegramChannel.  Only the HTTP endpoints differ.
    """

    name = "bale"
    display_name = "Bale"

    # Commands registered with Bale's command menu (same as Telegram).
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("stop", "Stop the current task"),
        BotCommand("restart", "Restart the bot"),
        BotCommand("status", "Show bot status"),
        BotCommand("history", "Show recent conversation messages"),
        BotCommand("goal", "Start a sustained objective (long-running task)"),
        BotCommand("pairing", "Manage DM pairing (approve/deny/list)"),
        BotCommand("model", "Switch runtime model preset"),
        BotCommand("skill", "List enabled skills"),
        BotCommand("help", "Show available commands"),
    ]

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return BaleConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = BaleConfig.model_validate(config)
        # Call TelegramChannel.__init__ directly (not super() which would
        # rebuild the config as TelegramConfig).
        TelegramChannel.__init__(self, config, bus)
        self.config: BaleConfig = config

    # ------------------------------------------------------------------
    # start() — identical to TelegramChannel.start() except for
    #           .base_url() / .base_file_url() on the Application builder.
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the Bale bot (long-polling or webhook)."""
        if not self.config.token:
            self.logger.error("bot token not configured")
            return

        self._running = True

        proxy = self.config.proxy or None

        # Separate pools so long-polling (getUpdates) never starves outbound sends.
        api_request = HTTPXRequest(
            connection_pool_size=self.config.connection_pool_size,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        poll_request = HTTPXRequest(
            connection_pool_size=4,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )

        builder = (
            Application.builder()
            .token(self.config.token)
            # ---- Bale-specific: route to Bale API servers ----
            .base_url(BALE_API_BASE_URL)
            .base_file_url(BALE_API_FILE_URL)
            # ----------------------------------------------------
            .request(api_request)
            .get_updates_request(poll_request)
        )
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers (using Regex to support @username suffixes before bot initialization)
        self._app.add_handler(MessageHandler(filters.Regex(r"^/start(?:@\w+)?$"), self._on_start))
        self._app.add_handler(
            MessageHandler(
                filters.Regex(TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE),
                self._forward_command,
            )
        )
        self._app.add_handler(
            MessageHandler(
                filters.Regex(
                    r"^/(dream-log|dream_log|dream-restore|dream_restore)(?:@\w+)?(?:\s+.*)?$"
                ),
                self._forward_command,
            )
        )
        self._app.add_handler(MessageHandler(filters.Regex(r"^/help(?:@\w+)?$"), self._on_help))

        # Add message handler for text, photos, video, voice, documents, and locations
        self._app.add_handler(
            MessageHandler(
                (
                    filters.TEXT
                    | filters.PHOTO
                    | filters.VIDEO
                    | filters.VIDEO_NOTE
                    | filters.ANIMATION
                    | filters.VOICE
                    | filters.AUDIO
                    | filters.Document.ALL
                    | filters.LOCATION
                )
                & ~filters.COMMAND,
                self._on_message,
            )
        )

        # Conditionally register inline keyboard callback handler
        if self.config.inline_keyboards:
            self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
            allowed_updates = ["message", "callback_query"]
            self.logger.debug("inline keyboards enabled")
        else:
            allowed_updates = ["message"]

        if self.config.mode == "webhook":
            self.logger.info("Starting bot (webhook mode)...")
        else:
            self.logger.info("Starting bot (polling mode)...")

        # Initialize and start receiving updates
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        self.logger.info("bot @{} connected", bot_info.username)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            self.logger.debug("bot commands registered")
        except Exception as e:
            self.logger.warning("Failed to register bot commands: {}", e)

        if self.config.mode == "webhook":
            await self._app.updater.start_webhook(
                listen=self.config.webhook_listen_host,
                port=self.config.webhook_listen_port,
                url_path=self.config.webhook_path.lstrip("/"),
                webhook_url=self.config.webhook_url.strip(),
                allowed_updates=allowed_updates,
                drop_pending_updates=False,
                secret_token=self.config.webhook_secret_token.strip(),
                max_connections=self.config.webhook_max_connections,
            )
        else:
            # Start polling (this runs until stopped)
            await self._app.updater.start_polling(
                allowed_updates=allowed_updates,
                drop_pending_updates=False,  # Process pending messages on startup
                error_callback=self._on_polling_error,
            )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)