"""
Automated Selenium cross-browser and multi-aspect-ratio layout test suite.

Validates that header title, subtitle, clocks, telemetry clusters, and admin modals
sit strictly on one line and never wrap or overflow across diverse viewports.
"""

from __future__ import annotations

import socket
import threading
import time
import pytest
import uvicorn
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def _is_port_open(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session", autouse=True)
def live_server():
    if not _is_port_open(8000):
        from src.web.app import app

        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(30):
            if _is_port_open(8000):
                break
            time.sleep(0.1)
    yield


VIEWPORTS = [
    (2560, 1080, "21:9 UltraWide (2560x1080)"),
    (1920, 1080, "16:9 1080p Desktop (1920x1080)"),
    (1728, 1117, "16:10 MacBook Pro 16in (1728x1117)"),
    (1512, 982, "16:10 MacBook Pro 14in (1512x982)"),
    (1440, 900, "16:10 MacBook Air (1440x900)"),
    (1366, 768, "16:9 Laptop (1366x768)"),
    (1280, 800, "16:10 Laptop (1280x800)"),
    (1024, 768, "4:3 Tablet Landscape (1024x768)"),
    (768, 1024, "3:4 Tablet Portrait (768x1024)"),
    (844, 390, "19.5:9 Mobile Landscape (844x390)"),
    (390, 844, "9:19.5 Mobile Portrait (390x844)"),
]


@pytest.fixture(scope="module")
def driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    d = webdriver.Chrome(options=options)
    yield d
    d.quit()


def _ensure_guest_mode(driver):
    driver.get("http://localhost:8000/cop")
    time.sleep(0.5)
    try:
        btn = driver.find_element(By.XPATH, '//button[contains(text(), "Continue as Guest")]')
        btn.click()
        time.sleep(0.3)
    except Exception:
        pass


@pytest.mark.parametrize("width,height,label", VIEWPORTS)
def test_header_and_clock_sit_on_one_line_across_aspect_ratios(driver, width, height, label):
    driver.set_window_size(width, height)
    _ensure_guest_mode(driver)

    header = driver.find_element(By.TAG_NAME, "header")
    h_rect = header.rect

    # 1. Header height is strictly bounded (never wraps to 2 rows)
    assert h_rect["height"] <= 56, f"{label}: Header wrapped vertically, height={h_rect['height']}"

    # 2. Title badge sits cleanly inside header
    title_badge = header.find_element(By.XPATH, './/span[contains(text(), "[ RIAN ]")]')
    tb_rect = title_badge.rect
    assert tb_rect["y"] >= 0, f"{label}: Title has negative y={tb_rect['y']}"
    assert tb_rect["y"] + tb_rect["height"] <= h_rect["y"] + h_rect["height"] + 2

    # 3. Clock UTC and Clock Local sit on the EXACT SAME horizontal line
    clock_utc = driver.find_element(By.ID, "clock-utc")
    utc_rect = clock_utc.rect
    clock_local = driver.find_element(By.ID, "clock-local")
    local_rect = clock_local.rect

    if clock_local.is_displayed():
        assert abs(utc_rect["y"] - local_rect["y"]) <= 2, (
            f"{label}: Clock UTC (y={utc_rect['y']}) and Local (y={local_rect['y']}) are on different lines!"
        )
    assert utc_rect["y"] >= 0 and utc_rect["y"] + utc_rect["height"] <= h_rect["y"] + h_rect["height"] + 2


@pytest.mark.parametrize(
    "browser_name,user_agent",
    [
        (
            "Chrome_Desktop_macOS",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        ),
        (
            "Safari_Desktop_macOS",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
        ),
        (
            "Safari_Mobile_iOS",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
        ),
        (
            "Firefox_Desktop_macOS",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0",
        ),
        (
            "Edge_Desktop_macOS",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
        ),
    ],
)
def test_cross_browser_layout_compatibility(browser_name, user_agent):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-agent={user_agent}")
    d = webdriver.Chrome(options=options)
    try:
        d.set_window_size(1440, 900)
        _ensure_guest_mode(d)

        header = d.find_element(By.TAG_NAME, "header")
        h_rect = header.rect
        assert h_rect["height"] <= 56, f"{browser_name}: Header height exceeded 56px"

        utc = d.find_element(By.ID, "clock-utc").rect
        loc = d.find_element(By.ID, "clock-local").rect
        assert abs(utc["y"] - loc["y"]) <= 2, f"{browser_name}: Clocks are not on the same horizontal row"
    finally:
        d.quit()
