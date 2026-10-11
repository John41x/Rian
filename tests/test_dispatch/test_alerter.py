"""Unit tests for Twilio tactical alerter debouncing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.dispatch.alerter import DEBOUNCE_SECONDS, TwilioAlerter


@pytest.mark.asyncio
async def test_mock_mode_dispatches(sample_directive):
    alerter = TwilioAlerter(account_sid="mock_sid")
    result = await alerter.send_sms(sample_directive)
    assert result.dispatched is True
    assert result.mock is True
    assert "[RIAN TACTICAL ALERT]" in result.message
    assert "INTERCEPT AT South Gate" in result.message


@pytest.mark.asyncio
async def test_debounce_blocks_duplicate_within_300_seconds(sample_directive, reference_time):
    alerter = TwilioAlerter(account_sid="mock_sid")
    first = await alerter.send_sms(sample_directive, now=reference_time)
    assert first.dispatched is True

    second = await alerter.send_sms(
        sample_directive,
        now=reference_time + timedelta(seconds=60),
    )
    assert second.dispatched is False
    assert second.skipped_reason is not None
    assert "debounced" in second.skipped_reason


@pytest.mark.asyncio
async def test_debounce_allows_resend_after_window(sample_directive, reference_time):
    alerter = TwilioAlerter(account_sid="mock_sid")
    await alerter.send_sms(sample_directive, now=reference_time)
    later = await alerter.send_sms(
        sample_directive,
        now=reference_time + timedelta(seconds=DEBOUNCE_SECONDS + 1),
    )
    assert later.dispatched is True


@pytest.mark.asyncio
async def test_debounce_allows_eta_shift_over_3_minutes(sample_directive, reference_time):
    alerter = TwilioAlerter(account_sid="mock_sid")
    await alerter.send_sms(sample_directive, now=reference_time)

    shifted = sample_directive.model_copy(
        update={
            "target_arrival_time": sample_directive.target_arrival_time
            + timedelta(minutes=4)
        }
    )
    result = await alerter.send_sms(
        shifted,
        now=reference_time + timedelta(seconds=30),
    )
    assert result.dispatched is True
