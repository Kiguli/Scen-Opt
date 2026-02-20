#!/usr/bin/env python3
"""Take screenshots of the web interface for documentation."""
import asyncio
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = "/Users/ben/Documents/Tools/ScenarioApproachTool/screenshots"

async def main():
    import os
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 1200})

        # Handle alert dialogs automatically
        page.on("dialog", lambda d: asyncio.ensure_future(d.accept()))

        await page.goto("http://127.0.0.1:5000/")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)

        # --- Load benchmark via Detect Program modal ---
        await page.click('button[data-target="#detectProgramModal"]')
        await asyncio.sleep(1)

        # Upload the benchmark JSON
        await page.set_input_files('#detect-program-file',
            "benchmarks/LP_half_width_2d/data/benchmark.json")
        await asyncio.sleep(0.5)

        # Click "Load" button
        await page.click('#detectProgramModal .btn-primary')
        await asyncio.sleep(1)

        # Set multiple rho and tau values for interesting graphs
        # First select the general formulation to show rho+tau fields
        await page.select_option('#lp-options', 'robust-regularization-relaxation')
        await asyncio.sleep(0.5)

        await page.fill('#lp-rho', '0,0.25,0.5,0.75,1')
        await page.fill('#lp-tau', '0,0.01,0.1,1')
        await asyncio.sleep(0.3)

        # Upload scenarios file
        await page.set_input_files('#file',
            "benchmarks/LP_half_width_2d/data/scenarios.csv")
        await asyncio.sleep(0.5)

        # Screenshot 1: Tool (everything above results, including Solve button)
        # Scroll to top first
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.5)

        # Find the Result card and use its top edge as the clip boundary
        result_card = page.locator('.card.mt-4:has(.card-title:has-text("Result"))').first
        result_box = await result_card.bounding_box()

        await page.screenshot(
            path=f"{SCREENSHOTS_DIR}/1_tool.png",
            clip={"x": 0, "y": 0, "width": 1400, "height": result_box["y"] - 5},
            full_page=True
        )
        print("Screenshot 1: Tool - done")

        # Now solve
        await page.click('button:has-text("Solve")')

        # Wait for results to appear
        await page.wait_for_selector('.result-table', timeout=120000)
        await asyncio.sleep(3)  # Let MathJax and charts render

        # Screenshot 2: Results table (full width of the result card)
        await page.click('#result-table-tab')
        await asyncio.sleep(0.5)
        result_card_el = page.locator('.card.mt-4:has(.card-title:has-text("Result"))').first
        await result_card_el.scroll_into_view_if_needed()
        await asyncio.sleep(0.5)
        await result_card_el.screenshot(path=f"{SCREENSHOTS_DIR}/2_results.png")
        print("Screenshot 2: Results table - done")

        # Screenshot 3: tau graph tab
        await page.click('a#result-graph2-tab')
        await asyncio.sleep(1.5)
        tau_pane = page.locator('#result-graph2')
        await tau_pane.screenshot(path=f"{SCREENSHOTS_DIR}/3_tau_graph.png")
        print("Screenshot 3: Tau graph - done")

        # Screenshot 4: rho graph tab
        await page.click('a#result-graph-tab')
        await asyncio.sleep(1.5)
        rho_pane = page.locator('#result-graph')
        await rho_pane.screenshot(path=f"{SCREENSHOTS_DIR}/4_rho_graph.png")
        print("Screenshot 4: Rho graph - done")

        await browser.close()
        print(f"\nAll screenshots saved to {SCREENSHOTS_DIR}/")

asyncio.run(main())
