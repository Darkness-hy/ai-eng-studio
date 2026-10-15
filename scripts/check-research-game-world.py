from __future__ import annotations

import argparse
import re
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def track_failures(page: Page, failures: list[str]) -> None:
    page.on(
        "console",
        lambda message: failures.append(f"console: {message.text}")
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: failures.append(f"pageerror: {error}"))
    page.on(
        "response",
        lambda response: failures.append(f"http {response.status}: {response.url}")
        if response.status >= 400
        else None,
    )


def assert_viewport(page: Page, failures: list[str], label: str) -> None:
    if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
        failures.append(f"{label}: horizontal overflow")
    broken_images = page.locator("img").evaluate_all(
        "images => images.filter(image => image.complete && image.naturalWidth === 0).length"
    )
    if broken_images:
        failures.append(f"{label}: {broken_images} broken images")
    canvas = page.locator("canvas").first
    canvas.wait_for(state="visible")
    box = canvas.bounding_box()
    if not box or box["width"] < 300 or box["height"] < 480:
        failures.append(f"{label}: invalid game canvas bounds {box}")
    screenshot_bytes = canvas.screenshot()
    if len(screenshot_bytes) < 10_000:
        failures.append(f"{label}: canvas appears blank ({len(screenshot_bytes)} screenshot bytes)")


def choose_avatar(page: Page, role: str) -> None:
    page.get_by_role("heading", name="选择你的科研角色").wait_for(state="visible")
    page.get_by_role("button", name=re.compile(role)).click()
    page.get_by_role("button", name="进入科研公社").click()
    page.get_by_role("heading", name="科研公社大厅", exact=True).wait_for(state="visible")
    page.locator("canvas").first.wait_for(state="visible")
    page.wait_for_timeout(1_200)


def click_world(page: Page, world_x: float, world_y: float) -> None:
    canvas = page.locator("canvas").first
    box = canvas.bounding_box()
    if not box:
        raise RuntimeError("game canvas has no bounding box")
    # Desktop checks use zoom=1. The 1600-wide world is centered in the canvas.
    screen_x = world_x - max((1600 - box["width"]) / 2, 0)
    canvas.click(position={"x": screen_x, "y": world_y})


def check_desktop(base_url: str, output: Path) -> None:
    failures: list[str] = []
    loaded_game_assets: set[str] = set()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-proxy-server"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        track_failures(page, failures)
        page.on(
            "response",
            lambda response: loaded_game_assets.add(response.url)
            if "/game/research-archipelago/" in response.url
            else None,
        )

        page.goto(f"{base_url}/research-game", wait_until="domcontentloaded")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="domcontentloaded")
        page.get_by_role("heading", name="选择你的科研角色").wait_for(state="visible")
        role_images = page.locator("img[alt$='角色']")
        if role_images.count() != 6:
            failures.append(f"avatar setup rendered {role_images.count()} role images instead of 6")
        role_sources = role_images.evaluate_all("images => images.map(image => image.src)")
        if len(set(role_sources)) != 6:
            failures.append("avatar setup role images are not six distinct assets")
        if role_images.evaluate_all(
            "images => images.filter(image => image.complete && image.naturalWidth === 0).length"
        ):
            failures.append("avatar setup contains a broken role image")
        page.screenshot(path=output / "desktop-avatar-setup.png", full_page=True)
        choose_avatar(page, "智能体设计师")
        assert_viewport(page, failures, "desktop lobby")
        canvas = page.locator("canvas").first
        ambient_before = canvas.screenshot()
        page.wait_for_timeout(700)
        ambient_after = canvas.screenshot()
        if ambient_before == ambient_after:
            failures.append("desktop lobby canvas is static while the player is idle")

        unexpectedly_eager = [
            name
            for name in (
                "model-system-observatory.webp",
                "personal-lab.webp",
                "compute-workshop.webp",
                "paper-review-workshop.webp",
                "academic-exchange-plaza.webp",
            )
            if any(name in url for url in loaded_game_assets)
        ]
        if unexpectedly_eager:
            failures.append(f"desktop lobby eagerly loaded scene assets: {unexpectedly_eager}")

        click_world(page, 790, 430)
        page.get_by_role("heading", name="林澈", exact=True).wait_for(state="visible")
        page.get_by_text("Agent 只提出建议和结构化动作", exact=False).wait_for(state="visible")
        page.screenshot(path=output / "desktop-campus-guide.png", full_page=True)
        page.get_by_role("button", name="关闭对话").click()

        page.keyboard.down("ArrowRight")
        page.locator("[data-player-moving='true']").wait_for(state="visible", timeout=3_000)
        page.wait_for_timeout(320)
        page.screenshot(path=output / "desktop-walking.png", full_page=True)
        page.keyboard.up("ArrowRight")
        page.locator("[data-player-moving='false']").wait_for(state="visible", timeout=3_000)
        page.locator("canvas").first.screenshot(path=output / "desktop-lobby-canvas.png")

        page.evaluate(
            """
            const key = 'research-game:player-profile-v1:guest';
            const profile = JSON.parse(localStorage.getItem(key));
            profile.currentScene = 'campus';
            profile.lastPosition = { x: 390, y: 330 };
            localStorage.setItem(key, JSON.stringify(profile));
            """
        )
        page.reload(wait_until="domcontentloaded")
        page.get_by_role("heading", name="科研公社大厅", exact=True).wait_for(state="visible")
        page.locator("canvas").first.wait_for(state="visible")
        page.wait_for_timeout(2_000)
        page.keyboard.down("ArrowUp")
        page.wait_for_url(f"{base_url}/research-game/academy", timeout=15_000)
        page.keyboard.up("ArrowUp")
        page.get_by_text("DESTINATION LOADED", exact=True).wait_for(state="visible", timeout=3_000)
        if page.locator("[data-current-scene='academy']").count() != 1:
            failures.append("portal route changed without loading the academy scene")
        academy_canvas = page.locator("canvas").first.screenshot()
        if academy_canvas == ambient_after:
            failures.append("academy scene canvas did not replace the lobby scene")
        # Reload at the destination so the fixed world-coordinate click below is
        # independent of the transition camera's in-flight interpolation.
        page.reload(wait_until="domcontentloaded")
        page.get_by_role("heading", name="科研训练馆", exact=True).wait_for(state="visible")
        page.wait_for_timeout(900)
        click_world(page, 800, 450)
        page.get_by_role("heading", name="苏老师", exact=True).wait_for(
            state="visible", timeout=12_000
        )
        page.get_by_text("503 LESSONS", exact=True).wait_for(state="visible", timeout=8_000)
        page.get_by_role("button", name="检索 503 节课程").wait_for(state="visible")
        page.screenshot(path=output / "desktop-research-mentor.png", full_page=True)
        page.get_by_role("button", name="关闭对话").click()

        page.goto(f"{base_url}/research-game/research", wait_until="domcontentloaded")
        page.get_by_role("heading", name="模型与系统实验室", exact=True).wait_for(state="visible")
        page.wait_for_timeout(900)
        click_world(page, 1080, 520)
        page.get_by_role("heading", name="项目与数据终端", exact=True).first.wait_for(
            state="visible", timeout=25_000
        )
        local_mode = page.get_by_text("本地演示模式不会启动真实实验", exact=False).count() > 0
        if local_mode:
            page.get_by_text("本地演示模式不会启动真实实验", exact=False).wait_for(
                state="visible"
            )
        else:
            page.wait_for_function(
                """
                () => document.querySelector('#research-template') ||
                  [...document.querySelectorAll('button')]
                    .some(button => button.textContent.includes('创建新的研究运行'))
                """,
                timeout=10_000,
            )
            create_another = page.get_by_role("button", name="创建新的研究运行")
            if create_another.count() > 0 and create_another.is_visible():
                create_another.click()
            selector = page.locator("#research-template")
            selector.wait_for(state="visible", timeout=8_000)
            if selector.locator("option").count() != 17:
                failures.append("connected project terminal does not contain 17 templates")
            page.get_by_role("button", name="创建 ResearchRun").click()
            page.get_by_role("button", name="创建新的研究运行").wait_for(
                state="visible", timeout=10_000
            )
        page.screenshot(path=output / "desktop-project-terminal.png", full_page=True)
        page.get_by_role("button", name="关闭科研工作台").click()

        page.evaluate(
            """
            const key = 'research-game:player-profile-v1:guest';
            const profile = JSON.parse(localStorage.getItem(key));
            profile.lastPosition = { x: 800, y: 720 };
            localStorage.setItem(key, JSON.stringify(profile));
            """
        )
        page.goto(f"{base_url}/research-game/research", wait_until="domcontentloaded")
        page.get_by_role("heading", name="模型与系统实验室", exact=True).wait_for(state="visible")
        page.wait_for_timeout(900)
        click_world(page, 500, 520)
        page.get_by_role("heading", name="研究假设白板", exact=True).first.wait_for(
            state="visible", timeout=25_000
        )
        if not local_mode:
            page.get_by_role("heading", name="提交可反驳的研究假设").wait_for(
                state="visible", timeout=8_000
            )
        page.screenshot(path=output / "desktop-hypothesis-board.png", full_page=True)

        if "[object Object]" in page.locator("body").inner_text():
            failures.append("desktop rendered [object Object]")
        browser.close()

    if failures:
        raise RuntimeError("\n".join(failures))


def check_mobile(base_url: str, output: Path) -> None:
    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-proxy-server"])
        page = browser.new_page(viewport={"width": 390, "height": 844})
        track_failures(page, failures)
        page.goto(f"{base_url}/research-game", wait_until="domcontentloaded")
        page.evaluate("localStorage.clear()")
        page.reload(wait_until="domcontentloaded")
        page.get_by_role("heading", name="选择你的科研角色").wait_for(state="visible")
        mobile_role_sources = page.locator("img[alt$='角色']").evaluate_all(
            "images => images.map(image => image.src)"
        )
        if len(set(mobile_role_sources)) != 6:
            failures.append("mobile avatar setup role images are not six distinct assets")
        page.screenshot(path=output / "mobile-avatar-setup.png", full_page=True)
        choose_avatar(page, "具身实践者")
        assert_viewport(page, failures, "mobile lobby")
        page.screenshot(path=output / "mobile-lobby.png", full_page=True)

        page.goto(f"{base_url}/research-game/academy", wait_until="domcontentloaded")
        page.get_by_role("heading", name="科研训练馆", exact=True).wait_for(state="visible")
        page.wait_for_timeout(1_000)
        assert_viewport(page, failures, "mobile academy")
        page.screenshot(path=output / "mobile-academy.png", full_page=True)
        if "[object Object]" in page.locator("body").inner_text():
            failures.append("mobile rendered [object Object]")
        browser.close()

    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the playable Research Island world")
    parser.add_argument("--url", default="http://127.0.0.1:5181")
    parser.add_argument("--output", default="output/research-game-world")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_url = args.url.rstrip("/")
    check_desktop(base_url, output)
    check_mobile(base_url, output)
    print(f"Playable world checks passed; screenshots: {output}")


if __name__ == "__main__":
    main()
