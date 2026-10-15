from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageStat
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright


PROFILE_KEY = "research-game:player-profile-v1:guest"
DISCOVERY_KEY = "research-game:field-notes-v1:guest"
SCENE_BOUNDS = {
    "campus": (-48, 48, -34, 34),
    "academy": (-11, 11, -9, 9),
    "research": (-12, 12, -9, 9),
    "compute": (-12, 12, -9, 9),
    "paper": (-12, 12, -9, 9),
    "plaza": (-13, 13, -10, 10),
}
WORKSPACES = {
    "research": ((3.6, -4.6), "使用项目与数据终端", "项目与数据终端"),
    "compute": ((3.4, 3.2), "操作离线实验集群", "离线实验集群"),
    "paper": ((3.8, -4.4), "打开分析与复现稿件", "分析与复现稿件"),
    "plaza": ((3.4, -3.8), "查看验证研究海报板", "验证研究海报板"),
}
SCENE_NAMES = {
    "campus": "科研公社大厅",
    "academy": "科研训练馆",
    "research": "模型与系统实验室",
    "compute": "算力工坊",
    "paper": "论文与评审工坊",
    "plaza": "学术交流广场",
}


def track_failures(page: Page, failures: list[str]) -> None:
    page.on(
        "console",
        lambda message: failures.append(f"console {message.type}: {message.text}")
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: failures.append(f"pageerror: {error}"))
    page.on(
        "response",
        lambda response: failures.append(f"HTTP {response.status}: {response.url}")
        if response.status >= 400
        else None,
    )


def seed_profile(
    page: Page,
    scene: str = "campus",
    x: float = 800,
    y: float = 570,
) -> None:
    page.evaluate(
        """
        ([key, scene, x, y]) => localStorage.setItem(key, JSON.stringify({
          avatarPreset: 'robotics-gold',
          currentScene: scene,
          lastPosition: { x, y },
          tutorialStep: 2,
        }))
        """,
        [PROFILE_KEY, scene, x, y],
    )


def seed_world_position(page: Page, scene: str, x: float, z: float) -> None:
    min_x, max_x, min_z, max_z = SCENE_BOUNDS[scene]
    legacy_x = (x - min_x) / (max_x - min_x) * 1600
    legacy_y = (z - min_z) / (max_z - min_z) * 900
    seed_profile(page, scene, legacy_x, legacy_y)


def seed_world_position_and_reload(page: Page, scene: str, x: float, z: float) -> None:
    min_x, max_x, min_z, max_z = SCENE_BOUNDS[scene]
    legacy_x = (x - min_x) / (max_x - min_x) * 1600
    legacy_y = (z - min_z) / (max_z - min_z) * 900
    try:
        page.evaluate(
            """
            ([key, scene, x, y]) => {
              localStorage.setItem(key, JSON.stringify({
                avatarPreset: 'robotics-gold',
                currentScene: scene,
                lastPosition: { x, y },
                tutorialStep: 2,
              }));
              window.location.reload();
            }
            """,
            [PROFILE_KEY, scene, legacy_x, legacy_y],
        )
    except PlaywrightError:
        pass
    page.wait_for_load_state("domcontentloaded")


def scene_url(base_url: str, scene: str) -> str:
    return f"{base_url}/research-game" if scene == "campus" else f"{base_url}/research-game/{scene}"


def open_scene(page: Page, base_url: str, scene: str) -> dict:
    print(f"checking 3D scene: {scene}", flush=True)
    page.goto(scene_url(base_url, scene), wait_until="domcontentloaded")
    seed_profile(page, scene, 800, 570 if scene == "campus" else 720)
    try:
        page.reload(wait_until="domcontentloaded")
    except PlaywrightError:
        page.goto(scene_url(base_url, scene), wait_until="domcontentloaded")
    diagnostics = wait_for_3d(page, scene)
    if scene != "campus":
        page.locator(".scene-arrival").wait_for(state="detached", timeout=4_000)
    return diagnostics


def wait_for_3d(page: Page, scene: str) -> dict:
    page.locator(f"[data-renderer='three'][data-current-scene='{scene}']").wait_for(
        state="visible", timeout=30_000
    )
    page.wait_for_function(
        """
        scene => window.__THREE_GAME_DIAGNOSTICS__?.sceneId === scene &&
          window.__THREE_GAME_DIAGNOSTICS__?.fps > 0
        """,
        arg=scene,
        timeout=30_000,
    )
    page.wait_for_function(
        "window.__THREE_GAME_DIAGNOSTICS__?.grounded === true",
        timeout=10_000,
    )
    return page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")


def assert_canvas_nonblank(
    page: Page,
    failures: list[str],
    label: str,
    output: Path,
) -> tuple[float, float, float]:
    canvas = page.locator("canvas").first
    canvas.wait_for(state="visible")
    box = canvas.bounding_box()
    if not box or box["width"] < 300 or box["height"] < 480:
        failures.append(f"{label}: invalid canvas bounds {box}")
        return (0, 0, 0)
    screenshot = canvas.screenshot(path=output / f"{label}-canvas.png")
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    sample = image.resize((160, 100))
    stat = ImageStat.Stat(sample)
    mean_stddev = sum(stat.stddev) / len(stat.stddev)
    colors = image.resize((120, 80)).getcolors(maxcolors=9_601) or []
    if mean_stddev < 12 or len(colors) < 24:
        failures.append(
            f"{label}: canvas appears blank (stddev={mean_stddev:.2f}, colors={len(colors)})"
        )
    return tuple(round(value, 2) for value in stat.mean)


def assert_viewport(page: Page, failures: list[str], label: str) -> None:
    if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
        failures.append(f"{label}: horizontal overflow")
    if "[object Object]" in page.locator("body").inner_text():
        failures.append(f"{label}: rendered [object Object]")


def distance(a: dict, b: dict, key: str) -> float:
    return math.sqrt(sum((a[key][axis] - b[key][axis]) ** 2 for axis in ("x", "y", "z")))


def click_ground(page: Page) -> None:
    canvas = page.locator("canvas").first
    box = canvas.bounding_box()
    if not box:
        raise RuntimeError("canvas has no bounds")
    page.mouse.click(box["x"] + box["width"] * 0.28, box["y"] + box["height"] * 0.68)


def check_click_navigation(page: Page, failures: list[str], output: Path) -> None:
    initial = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    page.keyboard.down("w")
    page.wait_for_timeout(600)
    page.keyboard.up("w")
    keyboard_after = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    if distance(initial, keyboard_after, "player") > 0.08:
        failures.append("desktop: movement keys still move the player")

    click_ground(page)
    page.wait_for_function(
        "window.__THREE_GAME_DIAGNOSTICS__?.navigationTarget !== null",
        timeout=5_000,
    )
    page.locator("[data-player-moving='true']").wait_for(state="visible", timeout=5_000)
    page.wait_for_timeout(950)
    moved = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    if distance(initial, moved, "player") < 0.45:
        failures.append("desktop: click destination did not move the player")
    if distance(initial, moved, "camera") < 0.25:
        failures.append("desktop: follow camera did not track click movement")
    page.screenshot(path=output / "desktop-click-navigation.png", full_page=True)
    page.keyboard.press("Escape")
    page.wait_for_function(
        "window.__THREE_GAME_DIAGNOSTICS__?.navigationTarget === null",
        timeout=5_000,
    )
    page.locator("[data-player-moving='false']").wait_for(state="visible", timeout=5_000)
    page.wait_for_timeout(200)

    before_orbit = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    canvas = page.locator("canvas").first.bounding_box()
    if canvas:
        center_x = canvas["x"] + canvas["width"] / 2
        center_y = canvas["y"] + canvas["height"] / 2
        page.mouse.move(center_x, center_y)
        page.mouse.down(button="right")
        page.mouse.move(center_x + 140, center_y + 45, steps=6)
        page.mouse.up(button="right")
        page.wait_for_timeout(450)
        after_orbit = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
        if distance(before_orbit, after_orbit, "camera") < 0.3:
            failures.append("desktop: right-drag did not orbit the camera")
        if distance(before_orbit, after_orbit, "player") > 0.08:
            failures.append("desktop: right-drag issued a movement command")


def check_ground_stability(page: Page, failures: list[str], output: Path) -> None:
    page.emulate_media(reduced_motion="reduce")
    seed_world_position(page, "campus", 0, 12)
    page.reload(wait_until="domcontentloaded")
    wait_for_3d(page, "campus")
    page.wait_for_timeout(6_000)
    canvas = page.locator("canvas").first
    before_bytes = canvas.screenshot(path=output / "desktop-ground-stability-before.png")
    page.wait_for_timeout(150)
    after_bytes = canvas.screenshot(path=output / "desktop-ground-stability-after.png")
    before = Image.open(io.BytesIO(before_bytes)).convert("RGB")
    after = Image.open(io.BytesIO(after_bytes)).convert("RGB")
    diff = ImageChops.difference(before, after)
    crop = diff.crop((220, 160, 1220, 560))
    mean_difference = sum(ImageStat.Stat(crop).mean) / 3
    red, green, blue = crop.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    changed_mask = max_channel.point(lambda value: 255 if value > 5 else 0)
    changed_fraction = changed_mask.histogram()[255] / (crop.width * crop.height)
    ImageEnhance.Contrast(diff).enhance(4).save(
        output / "desktop-ground-stability-diff.png"
    )
    if mean_difference > 0.5 or changed_fraction > 0.01:
        failures.append(
            "desktop campus: unstable ground pixels "
            f"(mean={mean_difference:.3f}, changed={changed_fraction:.3%})"
        )
    page.emulate_media(reduced_motion="no-preference")


def assert_dialogue_readability(
    dialogue,
    failures: list[str],
    label: str,
    minimum_portrait_height: float,
) -> None:
    message_font = dialogue.locator(".npc-dialogue-message").first.evaluate(
        "element => parseFloat(getComputedStyle(element).fontSize)"
    )
    input_font = dialogue.locator(".npc-dialogue-form input").first.evaluate(
        "element => parseFloat(getComputedStyle(element).fontSize)"
    )
    portrait = dialogue.locator(".npc-dialogue-portrait").bounding_box()
    if message_font < 14:
        failures.append(f"{label}: dialogue copy is too small ({message_font}px)")
    if input_font < 14:
        failures.append(f"{label}: dialogue input is too small ({input_font}px)")
    if not portrait or portrait["height"] < minimum_portrait_height:
        failures.append(f"{label}: portrait is too small ({portrait})")


def check_guide_dialogue(page: Page, failures: list[str], output: Path) -> None:
    seed_world_position(page, "campus", 0.8, 11.5)
    page.reload(wait_until="domcontentloaded")
    wait_for_3d(page, "campus")
    page.get_by_role("button", name="与向导林澈交谈").wait_for(
        state="visible", timeout=8_000
    )
    page.get_by_role("button", name="与向导林澈交谈").click()
    page.get_by_role("dialog", name="与林澈交谈").wait_for(state="visible", timeout=10_000)
    page.get_by_text("科研岛向导 Agent", exact=True).wait_for(state="visible")
    dialogue = page.get_by_role("dialog", name="与林澈交谈")
    dialogue.get_by_role("button", name="讨论科研路线").click()
    dialogue.locator(".npc-dialogue-conversation").wait_for(state="visible")
    dialogue.get_by_role("button", name="科研训练馆").wait_for(state="visible")
    dialogue.get_by_placeholder("向林澈继续提问…").wait_for(state="visible")
    if page.locator(".agent-panel").count() > 0:
        raise RuntimeError("guide dialogue escaped into the legacy page-like agent panel")
    assert_dialogue_readability(dialogue, failures, "desktop guide", 350)
    page.screenshot(path=output / "desktop-guide-conversation.png", full_page=True)
    dialogue.get_by_role("button", name="结束交谈").click()


def check_long_route_target(page: Page, failures: list[str]) -> None:
    initial = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    page.get_by_role("button", name=re.compile(r"^林澈")).click(force=True)
    page.wait_for_function(
        """
        () => {
          const target = window.__THREE_GAME_DIAGNOSTICS__?.navigationTarget;
          return target && Math.hypot(target.x - 0.8, target.z - 9.5) < 3;
        }
        """,
        timeout=5_000,
    )
    page.wait_for_timeout(2_500)
    moving = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    if distance(initial, moving, "player") < 0.3:
        failures.append("desktop campus: long route to guide did not advance")
    page.keyboard.press("Escape")
    page.wait_for_function(
        "window.__THREE_GAME_DIAGNOSTICS__?.navigationTarget === null", timeout=5_000
    )


def check_mentor_dialogue(page: Page, base_url: str, output: Path) -> None:
    page.goto(scene_url(base_url, "academy"), wait_until="domcontentloaded")
    seed_world_position(page, "academy", 0, -0.4)
    page.reload(wait_until="domcontentloaded")
    wait_for_3d(page, "academy")
    page.locator(".scene-arrival").wait_for(state="detached", timeout=4_000)
    page.get_by_role("button", name="与苏老师讨论科研方法").wait_for(
        state="visible", timeout=8_000
    )
    page.get_by_role("button", name="与苏老师讨论科研方法").click()
    dialogue = page.get_by_role("dialog", name="与苏老师交谈")
    dialogue.wait_for(state="visible")
    dialogue.get_by_role("button", name="开始科研训练").click()
    dialogue.locator(".npc-dialogue-mentor").wait_for(state="visible")
    dialogue.get_by_text("科研训练路径", exact=True).wait_for(state="visible")
    dialogue.get_by_placeholder("告诉苏老师你卡在科研的哪一步…").wait_for(
        state="visible"
    )
    dialogue.locator(".npc-dialogue-lessons a").first.wait_for(
        state="visible", timeout=12_000
    )
    page.screenshot(path=output / "desktop-mentor-conversation.png", full_page=True)
    dialogue.get_by_role("button", name="结束交谈").click()


def check_discovery(page: Page, base_url: str, output: Path) -> None:
    page.evaluate("key => localStorage.removeItem(key)", DISCOVERY_KEY)
    page.goto(scene_url(base_url, "campus"), wait_until="domcontentloaded")
    seed_world_position(page, "campus", -15, 20)
    page.reload(wait_until="domcontentloaded")
    wait_for_3d(page, "campus")
    page.get_by_role("button", name="查看开放数据亭").wait_for(
        state="visible", timeout=8_000
    )
    page.get_by_role("button", name="查看开放数据亭").click()
    dialogue = page.get_by_role("dialog", name="开放数据亭")
    dialogue.wait_for(state="visible")
    dialogue.get_by_text("NEW FIELD NOTE", exact=True).wait_for(state="visible")
    dialogue.get_by_label("科研手记进度 1/6").wait_for(state="visible")
    page.screenshot(path=output / "desktop-field-note.png", full_page=True)
    dialogue.get_by_role("button", name="继续探索").click()
    if page.locator("[data-discovered-count='1']").count() != 1:
        raise RuntimeError("field note discovery counter did not update")
    page.reload(wait_until="domcontentloaded")
    wait_for_3d(page, "campus")
    page.locator("[data-discovered-count='1']").wait_for(state="visible")


def check_world_map(page: Page, scene: str, output: Path) -> None:
    page.get_by_role("button", name="打开世界地图").first.click()
    page.locator("[data-camera-mode='map']").wait_for(state="visible")
    page.wait_for_function(
        "window.__THREE_GAME_DIAGNOSTICS__?.cameraMode === 'map' && "
        "window.__THREE_GAME_DIAGNOSTICS__.camera.y > 20"
    )
    diagnostics = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
    if scene == "campus" and diagnostics["drawCalls"] > 100:
        raise RuntimeError(
            f"campus world map render budget exceeded: {diagnostics['drawCalls']} calls"
        )
    page.locator(".game3d-map-label").first.wait_for(state="visible")
    if scene == "campus":
        if page.locator(".game3d-map-label--portal").count() != 5:
            raise RuntimeError("campus world map does not expose all five destinations")
        if page.locator(".game3d-map-label--landmark").count() != 6:
            raise RuntimeError("campus world map does not expose all six field-note landmarks")
    page.screenshot(path=output / f"desktop-{scene}-map.png", full_page=True)
    page.get_by_role("button", name="关闭世界地图").first.click()
    page.locator("[data-camera-mode='follow']").wait_for(state="visible")


def check_workspace(
    page: Page,
    base_url: str,
    scene: str,
    output: Path,
) -> None:
    (world_x, world_z), prompt, heading = WORKSPACES[scene]
    seed_world_position(page, scene, world_x, world_z)
    page.reload(wait_until="domcontentloaded")
    wait_for_3d(page, scene)
    page.get_by_role("button", name=prompt).wait_for(state="visible", timeout=8_000)
    page.get_by_role("button", name=prompt).click()
    page.get_by_role("heading", name=heading, exact=True).first.wait_for(
        state="visible", timeout=15_000
    )
    page.screenshot(path=output / f"desktop-{scene}-workspace.png", full_page=True)
    page.get_by_role("button", name="关闭科研工作台").click()
    if page.url != scene_url(base_url, scene):
        raise RuntimeError(f"{scene}: workspace changed route unexpectedly")


def check_desktop(base_url: str, output: Path) -> None:
    failures: list[str] = []
    scene_means: dict[str, tuple[float, float, float]] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--no-proxy-server", "--enable-webgl", "--use-angle=swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        track_failures(page, failures)

        campus = open_scene(page, base_url, "campus")
        scene_means["campus"] = assert_canvas_nonblank(
            page, failures, "desktop-campus", output
        )
        assert_viewport(page, failures, "desktop campus")
        if campus["drawCalls"] > 230 or campus["triangles"] > 300_000:
            failures.append(
                f"desktop campus: render budget exceeded ({campus['drawCalls']} calls, "
                f"{campus['triangles']} triangles)"
            )
        if abs(campus["footY"]) > 0.14:
            failures.append(f"desktop campus: avatar feet are not grounded (footY={campus['footY']})")
        check_click_navigation(page, failures, output)
        check_ground_stability(page, failures, output)
        seed_profile(page, "campus", 800, 807)
        page.reload(wait_until="domcontentloaded")
        wait_for_3d(page, "campus")
        check_long_route_target(page, failures)
        check_guide_dialogue(page, failures, output)
        check_discovery(page, base_url, output)
        check_world_map(page, "campus", output)

        check_mentor_dialogue(page, base_url, output)

        for scene in ("academy", "research", "compute", "paper", "plaza"):
            diagnostics = open_scene(page, base_url, scene)
            scene_means[scene] = assert_canvas_nonblank(
                page, failures, f"desktop-{scene}", output
            )
            assert_viewport(page, failures, f"desktop {scene}")
            page.get_by_text(SCENE_NAMES[scene], exact=True).first.wait_for(state="visible")
            page.locator(".game3d-agent-label").first.wait_for(state="visible")
            if not diagnostics["grounded"] or abs(diagnostics["footY"]) > 0.14:
                failures.append(
                    f"desktop {scene}: invalid ground contact "
                    f"(grounded={diagnostics['grounded']}, footY={diagnostics['footY']})"
                )
            check_world_map(page, scene, output)
            if scene in WORKSPACES:
                check_workspace(page, base_url, scene, output)

        rounded_means = {tuple(round(value / 8) for value in mean) for mean in scene_means.values()}
        if len(rounded_means) < 4:
            failures.append(f"scene canvases are insufficiently distinct: {scene_means}")

        page.goto(f"{base_url}/research-game?renderer=2d", wait_until="domcontentloaded")
        page.locator("[data-renderer='three']").wait_for(state="detached", timeout=30_000)
        page.locator("canvas").first.wait_for(state="visible", timeout=30_000)
        page.wait_for_timeout(1_000)
        page.screenshot(path=output / "desktop-2d-fallback.png", full_page=True)
        assert_viewport(page, failures, "desktop 2D fallback")
        browser.close()

    if failures:
        raise RuntimeError("\n".join(failures))


def check_mobile(base_url: str, output: Path) -> None:
    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--no-proxy-server", "--enable-webgl", "--use-angle=swiftshader"]
        )
        page = browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True)
        track_failures(page, failures)
        initial = open_scene(page, base_url, "campus")
        assert_canvas_nonblank(page, failures, "mobile-campus", output)
        assert_viewport(page, failures, "mobile campus")
        canvas = page.locator("canvas").first.bounding_box()
        if canvas:
            page.touchscreen.tap(
                canvas["x"] + canvas["width"] * 0.28,
                canvas["y"] + canvas["height"] * 0.62,
            )
            page.wait_for_function(
                "window.__THREE_GAME_DIAGNOSTICS__?.navigationTarget !== null",
                timeout=5_000,
            )
            page.wait_for_timeout(2_000)
            moved = page.evaluate("window.__THREE_GAME_DIAGNOSTICS__")
            if distance(initial, moved, "player") < 0.12:
                failures.append("mobile: tap-to-move did not move the player")
        page.screenshot(path=output / "mobile-tap-navigation.png", full_page=True)
        page.keyboard.press("Escape")
        page.get_by_role("button", name="打开世界地图").last.click()
        page.locator("[data-camera-mode='map']").wait_for(state="visible")
        page.locator(".game3d-map-label").first.wait_for(state="visible")
        page.screenshot(path=output / "mobile-world-map.png", full_page=True)
        assert_viewport(page, failures, "mobile map")
        page.get_by_role("button", name="关闭世界地图").first.click()
        seed_world_position_and_reload(page, "campus", 0.8, 11.5)
        wait_for_3d(page, "campus")
        page.get_by_role("button", name="与向导林澈交谈").wait_for(
            state="visible", timeout=8_000
        )
        page.get_by_role("button", name="与向导林澈交谈").click()
        dialogue = page.get_by_role("dialog", name="与林澈交谈")
        dialogue.wait_for(state="visible")
        dialogue.get_by_role("button", name="讨论科研路线").click()
        dialogue.locator(".npc-dialogue-conversation").wait_for(state="visible")
        assert_dialogue_readability(dialogue, failures, "mobile guide", 220)
        page.screenshot(path=output / "mobile-guide-conversation.png", full_page=True)
        assert_viewport(page, failures, "mobile guide dialogue")
        dialogue_box = dialogue.locator(".npc-dialogue-box").bounding_box()
        if not dialogue_box or dialogue_box["y"] < 0 or dialogue_box["height"] > 820:
            failures.append(f"mobile guide dialogue exceeds viewport: {dialogue_box}")
        browser.close()

    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Research Island 3D interactions")
    parser.add_argument("--url", default="http://127.0.0.1:5190")
    parser.add_argument("--output", default="output/research-game-3d")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_url = args.url.rstrip("/")
    check_desktop(base_url, output)
    check_mobile(base_url, output)
    print(f"Research Island 3D checks passed; screenshots: {output}")


if __name__ == "__main__":
    main()
