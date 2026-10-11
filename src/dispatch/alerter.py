"""
Twilio SMS / WhatsApp tactical alert dispatcher with debouncing (spec 04 §4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from src.common.config import settings
from src.common.schemas import EntityClass, TacticalDirective

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 300.0
ETA_CHANGE_REDISPATCH_SECONDS = 180.0  # 3 minutes


@dataclass
class _DebounceRecord:
    last_sent_at: datetime
    last_target_eta: datetime


@dataclass
class AlertResult:
    dispatched: bool
    channel: str
    message: str
    mock: bool = False
    skipped_reason: Optional[str] = None


class TwilioAlerter:
    """Formats and sends tactical directives; mock mode for tests and dev."""

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        dispatch_numbers: Optional[str] = None,
        debounce_store: Optional[Dict[Tuple[str, str], _DebounceRecord]] = None,
    ) -> None:
        self.account_sid = account_sid or settings.TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or settings.TWILIO_AUTH_TOKEN
        self.from_number = from_number or settings.TWILIO_FROM_NUMBER
        self.dispatch_numbers = dispatch_numbers or settings.DISPATCH_PHONE_NUMBERS
        self._debounce: Dict[Tuple[str, str], _DebounceRecord] = (
            debounce_store if debounce_store is not None else {}
        )
        self._mock_log: List[AlertResult] = []

    @property
    def is_mock_mode(self) -> bool:
        return self.account_sid == "mock_sid" or not self.account_sid

    def format_message(self, directive: TacticalDirective) -> str:
        target_min = max(
            0.0,
            (directive.target_arrival_time - datetime.now(timezone.utc)).total_seconds()
            / 60.0,
        )
        resp_min = max(
            0.0,
            (
                directive.interceptor_arrival_time - datetime.now(timezone.utc)
            ).total_seconds()
            / 60.0,
        )
        margin_min = directive.safety_margin_seconds / 60.0
        target_eta_fmt = directive.target_arrival_time.strftime("%H:%M UTC")
        resp_eta_fmt = directive.interceptor_arrival_time.strftime("%H:%M UTC")
        return (
            f"[RIAN TACTICAL ALERT]\n"
            f"Target: {directive.target_class.value.upper()}\n"
            f"Track: {directive.track_code}\n"
            f"Directive: INTERCEPT AT {directive.chokepoint_name}\n"
            f"Target ETA: {target_eta_fmt} ({target_min:.1f} min)\n"
            f"Your ETA: {resp_eta_fmt} ({resp_min:.1f} min)\n"
            f"Tactical Margin: {margin_min:.1f} min advantage\n"
            f"Map: https://rian.domain/cop?track={directive.track_id}"
        )

    def _should_dispatch(
        self,
        track_id: UUID,
        chokepoint_id: UUID,
        target_eta: datetime,
        now: datetime,
    ) -> Tuple[bool, Optional[str]]:
        key = (str(track_id), str(chokepoint_id))
        record = self._debounce.get(key)
        if record is None:
            return True, None
        elapsed = (now - record.last_sent_at).total_seconds()
        if elapsed >= DEBOUNCE_SECONDS:
            return True, None
        eta_delta = abs((target_eta - record.last_target_eta).total_seconds())
        if eta_delta > ETA_CHANGE_REDISPATCH_SECONDS:
            return True, None
        return False, f"debounced ({elapsed:.0f}s since last alert)"

    def _parse_recipients(self) -> List[str]:
        return [n.strip() for n in self.dispatch_numbers.split(",") if n.strip()]

    async def send_sms(
        self,
        directive: TacticalDirective,
        now: Optional[datetime] = None,
    ) -> AlertResult:
        return await self._send(directive, channel="sms", now=now)

    async def send_whatsapp(
        self,
        directive: TacticalDirective,
        now: Optional[datetime] = None,
    ) -> AlertResult:
        return await self._send(directive, channel="whatsapp", now=now)

    async def _send(
        self,
        directive: TacticalDirective,
        channel: str,
        now: Optional[datetime] = None,
    ) -> AlertResult:
        now = now or datetime.now(timezone.utc)
        message = self.format_message(directive)
        allowed, reason = self._should_dispatch(
            directive.track_id,
            directive.target_chokepoint_id,
            directive.target_arrival_time,
            now,
        )
        if not allowed:
            result = AlertResult(
                dispatched=False,
                channel=channel,
                message=message,
                mock=self.is_mock_mode,
                skipped_reason=reason,
            )
            return result

        if self.is_mock_mode:
            result = AlertResult(
                dispatched=True,
                channel=channel,
                message=message,
                mock=True,
            )
            self._mock_log.append(result)
            logger.info("Mock %s alert to %s", channel, self._parse_recipients())
            self._record_debounce(directive, now)
            return result

        try:
            from twilio.rest import Client  # type: ignore[import-untyped]
        except ImportError:
            logger.error("twilio package not installed; falling back to mock")
            return await self._send_mock_fallback(directive, channel, message, now)

        client = Client(self.account_sid, self.auth_token)
        recipients = self._parse_recipients()
        for to in recipients:
            if channel == "whatsapp":
                client.messages.create(
                    from_=f"whatsapp:{self.from_number}",
                    to=f"whatsapp:{to}",
                    body=message,
                )
            else:
                client.messages.create(
                    from_=self.from_number,
                    to=to,
                    body=message,
                )
        self._record_debounce(directive, now)
        return AlertResult(dispatched=True, channel=channel, message=message, mock=False)

    async def _send_mock_fallback(
        self,
        directive: TacticalDirective,
        channel: str,
        message: str,
        now: datetime,
    ) -> AlertResult:
        result = AlertResult(dispatched=True, channel=channel, message=message, mock=True)
        self._mock_log.append(result)
        self._record_debounce(directive, now)
        return result

    def _record_debounce(self, directive: TacticalDirective, now: datetime) -> None:
        key = (str(directive.track_id), str(directive.target_chokepoint_id))
        self._debounce[key] = _DebounceRecord(
            last_sent_at=now,
            last_target_eta=directive.target_arrival_time,
        )
