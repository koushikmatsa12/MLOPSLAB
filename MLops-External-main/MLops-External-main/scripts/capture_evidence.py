"""Capture genuine live UI screenshots of MLflow Tracking/Registry and FastAPI serving.

Uses Playwright headless Chromium against locally started MLflow and Uvicorn servers,
saving crisp PNG evidence into docs/figures/ for inclusion in the PDF report.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from src.config import FIGURES_DIR, MLFLOW_TRACKING_URI, ROOT_DIR


def wait_for_http(url: str, timeout: float = 35.0) -> None:
    """Wait until an HTTP endpoint responds with status < 500."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Service at {url} did not become ready within {timeout}s")


def capture_all_screenshots() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mlflow_port = 5055
    api_port = 8055

    mlflow_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            MLFLOW_TRACKING_URI,
            "--host",
            "127.0.0.1",
            "--port",
            str(mlflow_port),
        ],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_http(f"http://127.0.0.1:{mlflow_port}/")
        wait_for_http(f"http://127.0.0.1:{api_port}/health")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1360, "height": 840})
            page = context.new_page()

            # 1. MLflow Experiments & Runs UI
            page.goto(
                f"http://127.0.0.1:{mlflow_port}/#/experiments/1/runs",
                wait_until="domcontentloaded",
            )
            page.wait_for_timeout(4000)
            # Switch to Model training view if visible & dismiss MLflow Assistant sidebar for wide table view
            try:
                mt_btn = page.get_by_text("Model training", exact=True)
                if mt_btn.count() > 0:
                    mt_btn.first.click()
                    page.wait_for_timeout(1500)
            except Exception:
                pass
            try:
                close_btn = page.locator("button[aria-label='Close'], button[aria-label='Close assistant']")
                if close_btn.count() > 0:
                    close_btn.first.click()
                    page.wait_for_timeout(800)
                else:
                    # Fallback: click the top-right 'x' button inside MLflow Assistant panel
                    page.mouse.click(1325, 36)
                    page.wait_for_timeout(800)
            except Exception:
                pass
            # Wait for the actual runs table rows to render
            try:
                page.get_by_text("RandomForest_Comparison").first.wait_for(timeout=15000)
                page.wait_for_timeout(1500)
            except Exception:
                page.wait_for_timeout(5000)
            page.screenshot(path=str(FIGURES_DIR / "mlflow_experiments_ui.png"), full_page=False)

            # 2. MLflow Registered Models UI
            page.goto(
                f"http://127.0.0.1:{mlflow_port}/#/models/WaterShortagePredictor",
                wait_until="domcontentloaded",
            )
            try:
                page.get_by_text("production").first.wait_for(timeout=12000)
                page.wait_for_timeout(1200)
            except Exception:
                page.wait_for_timeout(4000)
            page.screenshot(path=str(FIGURES_DIR / "mlflow_registry_ui.png"), full_page=False)

            # 3. FastAPI Swagger UI (/docs)
            page.goto(f"http://127.0.0.1:{api_port}/docs", wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            # Expand POST /predict block if present
            predict_block = page.locator("#operations-default-predict_shortage_predict_post")
            if predict_block.count() > 0:
                predict_block.click()
                page.wait_for_timeout(1000)
            page.screenshot(path=str(FIGURES_DIR / "fastapi_swagger_ui.png"), full_page=False)

            # 4. FastAPI Interactive UI — Shortage Risk scenario
            page.goto(f"http://127.0.0.1:{api_port}/ui", wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            page.click("button.btn-danger")
            page.wait_for_timeout(1200)
            page.screenshot(path=str(FIGURES_DIR / "api_shortage_risk.png"), full_page=False)

            # 5. FastAPI Interactive UI — Adequate Reserve scenario
            page.click("button.btn-safe")
            page.wait_for_timeout(1200)
            page.screenshot(path=str(FIGURES_DIR / "api_adequate_reserve.png"), full_page=False)

            # 6. FastAPI Interactive UI — Capacity Validation Error (> 100,000 L)
            page.fill("#tank_level_8pm", "105000")
            page.click("button.btn-primary")
            page.wait_for_timeout(1200)
            page.screenshot(path=str(FIGURES_DIR / "api_capacity_validation.png"), full_page=False)

            browser.close()
        print(f"Captured all MLflow and FastAPI screenshots in {FIGURES_DIR}")
    finally:
        mlflow_proc.terminate()
        api_proc.terminate()
        try:
            mlflow_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mlflow_proc.kill()
        try:
            api_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_proc.kill()


if __name__ == "__main__":
    capture_all_screenshots()

