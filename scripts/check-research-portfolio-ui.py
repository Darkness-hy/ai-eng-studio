from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def check_viewport(base_url: str, output: Path, name: str, width: int, height: int) -> None:
    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-proxy-server"])
        page = browser.new_page(viewport={"width": width, "height": height})
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

        page.goto(f"{base_url}/research-game", wait_until="domcontentloaded")
        page.get_by_role("heading", name="科研岛 · 个人 AI 导师：贝叶斯知识追踪").wait_for(
            state="visible"
        )
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            failures.append("research map has horizontal overflow")
        page.screenshot(path=output / f"{name}-map.png", full_page=True)

        page.goto(f"{base_url}/research-game/observatory", wait_until="domcontentloaded")
        page.get_by_role("button", name="创建新的研究运行").wait_for(state="visible")
        if "CodeSearchNet" in page.locator("header").nth(1).inner_text():
            failures.append("observatory header leaked the old CodeSearchNet subtitle")
        page.get_by_role("button", name="创建新的研究运行").click()
        selector = page.locator("#research-template")
        selector.wait_for(state="visible")
        if selector.locator("option").count() != 17:
            failures.append("template selector does not contain 17 options")
        selector.select_option("mcp_registry_governance_v1")
        page.get_by_text("治理决策准确率", exact=True).wait_for(state="visible")
        if "[object Object]" in page.locator("body").inner_text():
            failures.append("page rendered [object Object]")
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            failures.append("observatory has horizontal overflow")
        broken_images = page.locator("img").evaluate_all(
            "images => images.filter(image => image.complete && image.naturalWidth === 0).length"
        )
        if broken_images:
            failures.append(f"observatory has {broken_images} broken images")
        page.screenshot(path=output / f"{name}-observatory.png", full_page=True)

        page.goto(f"{base_url}/research-game/compute", wait_until="domcontentloaded")
        page.get_by_text("个人 AI 导师：贝叶斯知识追踪离线基准", exact=True).wait_for(
            state="visible"
        )
        page.get_by_text("干预后掌握度", exact=True).first.wait_for(state="visible")
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            failures.append("compute workspace has horizontal overflow")
        page.screenshot(path=output / f"{name}-compute.png", full_page=True)
        browser.close()

    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the connected zero-API capstone UI")
    parser.add_argument("--url", default="http://127.0.0.1:5181")
    parser.add_argument("--output", default="output/research-game-zero-api")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    check_viewport(args.url.rstrip("/"), output, "desktop", 1440, 1000)
    check_viewport(args.url.rstrip("/"), output, "mobile", 390, 844)
    print(f"UI checks passed; screenshots: {output}")


if __name__ == "__main__":
    main()
