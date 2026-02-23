"""Telegram Bot API helpers — zero dependencies beyond stdlib."""

from __future__ import annotations

import json
import logging
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger("dispatcher")


class TelegramClient:
    """Thin wrapper around Telegram Bot API using urllib (no deps)."""

    def __init__(self, bot_token: str, chat_id: int):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._api_base = f"https://api.telegram.org/bot{bot_token}"

    def request(self, method: str, payload: dict, timeout: int = 40) -> dict:
        body = json.dumps(payload).encode()
        req = Request(
            f"{self._api_base}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError) as exc:
            log.error("tg_request %s failed: %s", method, exc)
            return {}

    def send(
        self,
        text: str,
        reply_to: int | None = None,
        parse_mode: str | None = None,
        reply_markup: dict | None = None,
    ) -> int | None:
        """Send a message. Returns the new message_id or None."""
        data: dict = {"chat_id": self.chat_id, "text": text[:4096]}
        if reply_to:
            data["reply_parameters"] = {"message_id": reply_to}
        if parse_mode:
            data["parse_mode"] = parse_mode
        if reply_markup:
            data["reply_markup"] = reply_markup
        result = self.request("sendMessage", data)
        return result.get("result", {}).get("message_id")

    def edit(
        self,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict | None = None,
    ) -> bool:
        """Edit an existing message. Returns True on success."""
        data: dict = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text[:4096],
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        if reply_markup:
            data["reply_markup"] = reply_markup
        result = self.request("editMessageText", data)
        return bool(result.get("ok"))

    def typing(self):
        """Send 'typing...' indicator — lasts ~5s on client side."""
        self.request(
            "sendChatAction",
            {"chat_id": self.chat_id, "action": "typing"},
            timeout=5,
        )

    def react(self, message_id: int, emoji: str) -> bool:
        """Set a reaction emoji on a message."""
        result = self.request("setMessageReaction", {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "reaction": [{"type": "emoji", "emoji": emoji}],
        })
        return bool(result.get("ok"))

    def delete_message(self, message_id: int) -> bool:
        """Delete a message."""
        result = self.request("deleteMessage", {
            "chat_id": self.chat_id,
            "message_id": message_id,
        })
        return bool(result.get("ok"))

    def answer_callback(self, callback_query_id: str, text: str = "") -> bool:
        """Answer a callback query (inline keyboard button press)."""
        data: dict = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text
        result = self.request("answerCallbackQuery", data)
        return bool(result.get("ok"))

    def get_file_url(self, file_id: str) -> str | None:
        """Get download URL for a file by its file_id."""
        result = self.request("getFile", {"file_id": file_id})
        file_path = result.get("result", {}).get("file_path")
        if file_path:
            return f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        return None

    def download_file(self, file_id: str, dest_path: str) -> bool:
        """Download a Telegram file to a local path. Streams in chunks to avoid memory spikes."""
        url = self.get_file_url(file_id)
        if not url:
            return False
        try:
            req = Request(url)
            resp = urlopen(req, timeout=60)
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except (URLError, OSError) as exc:
            log.error("download_file failed: %s", exc)
            return False

    def set_my_commands(self, commands: list[dict]) -> bool:
        """Register bot commands menu. Each dict: {"command": "...", "description": "..."}."""
        result = self.request("setMyCommands", {"commands": commands})
        return bool(result.get("ok"))

    def _send_multipart(
        self,
        method: str,
        field_name: str,
        file_path: str,
        caption: str = "",
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> int | None:
        """Send a file via multipart/form-data. Returns message_id or None."""
        import mimetypes
        boundary = "----DispatcherBoundary"
        parts = []
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"chat_id\"\r\n\r\n"
            f"{self.chat_id}"
        )
        if reply_to:
            parts.append(
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"reply_parameters\"\r\n\r\n"
                f"{{\"message_id\": {reply_to}}}"
            )
        if caption:
            parts.append(
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"caption\"\r\n\r\n"
                f"{caption[:1024]}"
            )
        if parse_mode:
            parts.append(
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"parse_mode\"\r\n\r\n"
                f"{parse_mode}"
            )

        fname = file_path.rsplit("/", 1)[-1]
        mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        with open(file_path, "rb") as f:
            file_data = f.read()

        file_header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{fname}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n"
        )

        body = b""
        for p in parts:
            body += p.encode() + b"\r\n"
        body += file_header.encode()
        body += file_data + f"\r\n--{boundary}--\r\n".encode()

        req = Request(
            f"{self._api_base}/{method}",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            resp = urlopen(req, timeout=60)
            result = json.loads(resp.read())
            return result.get("result", {}).get("message_id")
        except Exception as exc:
            log.error("%s failed: %s", method, exc)
            return None

    def send_photo(
        self,
        photo_path: str,
        caption: str = "",
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> int | None:
        """Send a photo file. Returns the new message_id or None."""
        return self._send_multipart("sendPhoto", "photo", photo_path, caption, reply_to, parse_mode)

    def send_document(
        self,
        file_path: str,
        caption: str = "",
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> int | None:
        """Send a document/file. Returns the new message_id or None."""
        return self._send_multipart("sendDocument", "document", file_path, caption, reply_to, parse_mode)

    def poll(self, offset: int, poll_timeout: int = 30) -> list[dict]:
        data: dict = {
            "timeout": poll_timeout,
            "allowed_updates": ["message", "edited_message", "callback_query"],
        }
        if offset:
            data["offset"] = offset
        result = self.request("getUpdates", data, timeout=poll_timeout + 10)
        return result.get("result", [])
