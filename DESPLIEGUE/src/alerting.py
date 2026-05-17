import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_COOLDOWN_SECONDS = 300


class AlertManager:
    def __init__(
        self,
        history_path,
        *,
        webhook_url=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
    ):
        self.history_path = Path(history_path)
        self.webhook_url = webhook_url or None
        self.telegram_bot_token = telegram_bot_token or None
        self.telegram_chat_id = telegram_chat_id or None
        self.cooldown_seconds = int(cooldown_seconds)
        self.last_sent_at = {}
        self.active_alerts = {}

    @classmethod
    def from_env(cls, default_history_path):
        return cls(
            os.getenv("ALERT_HISTORY_PATH", default_history_path),
            webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", str(DEFAULT_COOLDOWN_SECONDS))),
        )

    def channels(self):
        channels = ["local_history"]
        if self.webhook_url:
            channels.append("webhook")
        if self.telegram_bot_token and self.telegram_chat_id:
            channels.append("telegram")
        return channels

    def trigger(self, key, category, severity, title, detail, metadata=None):
        now = time.time()
        previous = self.last_sent_at.get(key)
        if previous is not None and now - previous < self.cooldown_seconds:
            self.active_alerts[key] = {
                "category": category,
                "severity": severity,
                "title": title,
                "detail": detail,
                "metadata": metadata or {},
                "last_seen_at": datetime.now(timezone.utc).isoformat(),
                "cooldown_active": True,
            }
            return None

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "category": category,
            "severity": severity,
            "title": title,
            "detail": detail,
            "metadata": metadata or {},
            "channels": self.channels(),
        }

        delivery_errors = []
        self._write_local(event)

        try:
            self._send_webhook(event)
        except Exception as exc:
            delivery_errors.append(f"webhook: {exc}")

        try:
            self._send_telegram(event)
        except Exception as exc:
            delivery_errors.append(f"telegram: {exc}")

        if delivery_errors:
            event["delivery_errors"] = delivery_errors
            self._write_local({**event, "event_type": "delivery_error"})

        self.last_sent_at[key] = now
        self.active_alerts[key] = {
            "category": category,
            "severity": severity,
            "title": title,
            "detail": detail,
            "metadata": metadata or {},
            "last_seen_at": event["timestamp"],
            "cooldown_active": False,
        }
        return event

    def resolve(self, key):
        self.active_alerts.pop(key, None)

    def reset_runtime(self):
        self.last_sent_at.clear()
        self.active_alerts.clear()

    def recent(self, limit=20):
        if not self.history_path.is_file():
            return []

        with self.history_path.open("r", encoding="utf-8") as f:
            rows = [line.strip() for line in f if line.strip()]

        return [json.loads(row) for row in rows[-limit:]][::-1]

    def status(self, limit=20):
        return {
            "history_path": str(self.history_path),
            "channels": self.channels(),
            "cooldown_seconds": self.cooldown_seconds,
            "active_alerts": self.active_alerts,
            "recent_alerts": self.recent(limit=limit),
        }

    def _write_local(self, event):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _send_webhook(self, event):
        if not self.webhook_url:
            return

        payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()

    def _send_telegram(self, event):
        if not (self.telegram_bot_token and self.telegram_chat_id):
            return

        text = (
            f"[{event['severity'].upper()}] {event['title']}\n"
            f"Categoria: {event['category']}\n"
            f"Detalle: {event['detail']}"
        )
        data = urllib.parse.urlencode(
            {"chat_id": self.telegram_chat_id, "text": text}
        ).encode("utf-8")
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        request = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
