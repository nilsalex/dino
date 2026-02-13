"""Game interface using Playwright for keyboard control."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from playwright.async_api import async_playwright  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page, Playwright


class PlaywrightGameInterface:
    """Interface for executing actions in the game using Playwright."""

    def __init__(
        self,
        url: str,
        action_keys: list[str],
        browser_type: str = "firefox",
        cdp_port: int | None = None,
    ):
        self.url = url
        self.action_keys = action_keys
        self.browser_type = browser_type
        self.cdp_port = cdp_port
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._connected_via_cdp = False

    def _run_async(self, coro):
        """Run async coroutine in the event loop."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    async def _start_async(self) -> None:
        """Start the browser and navigate to the game URL."""
        playwright = await async_playwright().start()
        self._playwright = playwright

        if self.cdp_port is not None:
            self._browser = await playwright.chromium.connect_over_cdp(f"http://localhost:{self.cdp_port}")
            self._connected_via_cdp = True

            contexts = self._browser.contexts
            if not contexts:
                raise RuntimeError("No browser contexts found")

            pages = contexts[0].pages
            if not pages:
                raise RuntimeError("No pages found in browser context")

            self._page = pages[0]
            print(f"Connected to existing browser via CDP (port {self.cdp_port})")
        else:
            if self.browser_type == "firefox":
                self._browser = await playwright.firefox.launch(headless=False)
            elif self.browser_type == "chromium":
                self._browser = await playwright.chromium.launch(headless=False)
            else:
                raise ValueError(f"Unknown browser type: {self.browser_type}")

            if self._browser is None:
                raise RuntimeError("Browser failed to initialize")

            self._page = await self._browser.new_page()
            if self._page is None:
                raise RuntimeError("Failed to create page")
            await self._page.goto(self.url)  # type: ignore[union-attr]

    async def _execute_action_async(self, action: int) -> None:
        """Execute an action in the game.

        Args:
            action: The action index to execute (maps to action_keys).
        """
        if action >= len(self.action_keys):
            raise ValueError(f"Action {action} out of range (0-{len(self.action_keys) - 1})")

        key = self.action_keys[action]
        if key != "":
            if self._page is None:
                raise RuntimeError("Page not initialized")
            await self._page.keyboard.press(key)

    async def _reset_game_async(self) -> None:
        """Reset the game by pressing space."""
        if self._page is None:
            raise RuntimeError("Page not initialized")
        await self._page.keyboard.press(" ")

    async def _close_async(self) -> None:
        """Close the game interface."""
        if self._connected_via_cdp:
            if self._playwright:
                await self._playwright.stop()
        else:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()

    def start(self) -> None:
        """Start the browser and navigate to the game URL."""
        self._run_async(self._start_async())

    def execute_action(self, action: int) -> None:
        """Execute an action in the game.

        Args:
            action: The action index to execute (maps to action_keys).
        """
        self._run_async(self._execute_action_async(action))

    def reset_game(self) -> None:
        """Reset the game by pressing space."""
        self._run_async(self._reset_game_async())

    def close(self) -> None:
        """Close the game interface."""
        try:
            self._run_async(self._close_async())
        finally:
            if self._loop:
                self._loop.close()
            self._executor.shutdown(wait=True)
