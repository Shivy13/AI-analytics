"""Browser smoke test: drives the real UI in Chromium and fails on any
console error, missing chart, or broken interaction.

Usage: python3 tests/ui_smoke.py [base_url]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
SHOTS = Path(__file__).parent.parent / "shots"
SHOTS.mkdir(exist_ok=True)

errors: list[str] = []
checks: list[tuple[bool, str]] = []


def check(ok, label):
    checks.append((bool(ok), label))
    print(("  PASS  " if ok else "  FAIL  ") + label)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 1000})
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
              if m.type in ("error",) else None)
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        print("== landing ==")
        pg.goto(BASE, wait_until="networkidle")
        check(pg.locator("#dropzone").is_visible(), "dropzone visible")
        check(pg.locator(".sample").count() >= 3, "sample cards rendered")
        check("Local engine" in pg.locator("#engineText").inner_text(), "engine chip shows local mode")
        pg.screenshot(path=SHOTS / "1-landing.png", full_page=True)

        print("== load sample ==")
        pg.locator(".sample", has_text="Retail sales").first.click()
        pg.wait_for_selector("#kpiRow .kpi", timeout=30000)
        pg.wait_for_selector("#insightList .insight", timeout=30000)
        pg.wait_for_timeout(1200)
        check(pg.locator("#kpiRow .kpi").count() >= 4, f"KPI cards: {pg.locator('#kpiRow .kpi').count()}")
        n_ins = pg.locator("#insightList .insight").count()
        check(n_ins >= 8, f"insight cards: {n_ins}")
        n_charts = pg.locator("#autoCharts .chart-card svg").count()
        check(n_charts >= 4, f"auto charts rendered as SVG: {n_charts}")
        rects = pg.locator("#autoCharts svg rect.mark, #autoCharts svg path").count()
        check(rects > 10, f"chart marks drawn: {rects}")
        check(pg.locator("#datasetChip").is_visible(), "dataset chip visible")
        pg.screenshot(path=SHOTS / "2-overview.png", full_page=True)

        print("== insight actions ==")
        first_q = pg.locator('#insightList .insight [data-act="ask"]').first
        q_text = first_q.inner_text().replace("Ask: ", "")
        first_q.click()
        pg.wait_for_selector("#askLog .msg .answer-text", timeout=30000)
        pg.wait_for_timeout(900)
        ans = pg.locator("#askLog .msg .answer-text").first.inner_text()
        check(len(ans) > 40, f"insight 'Ask' produced an answer ({len(ans)} chars)")
        pg.screenshot(path=SHOTS / "3-ask-from-insight.png", full_page=True)

        print("== ask tab ==")
        pg.locator('#tabAsk input#askInput').fill("average profit by channel")
        pg.locator("#btnAsk").click()
        pg.wait_for_timeout(1500)
        answers = pg.locator("#askLog .msg")
        check(answers.count() >= 2, f"chat messages: {answers.count()}")
        newest = answers.first
        check(newest.locator("svg").count() >= 1, "answer includes a chart")
        check(newest.locator(".parse-chips .chip").count() >= 3, "parse chips shown")
        txt = newest.locator(".answer-text").inner_text().lower()
        check("channel" in txt or "profit" in txt, "answer mentions the asked slots")
        check(newest.locator(".suggest button").count() >= 1, "follow-up suggestions offered")
        pg.screenshot(path=SHOTS / "4-ask.png", full_page=True)

        print("== explore tab ==")
        pg.locator('.tab[data-tab="explore"]').click()
        pg.wait_for_timeout(1500)
        check(pg.locator("#exploreChart svg").count() >= 1, "explore chart rendered")
        check(pg.locator("#exploreTable table tbody tr").count() > 0, "explore table has rows")
        pg.select_option("#ctlDimension", "region")
        pg.select_option("#ctlMeasure", "revenue")
        pg.select_option("#ctlAgg", "avg")
        pg.wait_for_timeout(1200)
        check("Average revenue" in pg.locator("#exploreChart .chart-title").inner_text(),
              "chart title follows the controls")
        pg.locator("#btnAddFilter").click()
        pg.wait_for_timeout(300)
        pg.locator(".filter-row select[data-k='op']").first.select_option("gt")
        pg.locator(".filter-row input[data-k='value']").first.fill("5000")
        pg.locator(".filter-row input[data-k='value']").first.press("Enter")
        pg.wait_for_timeout(1200)
        meta = pg.locator("#exploreMeta").inner_text()
        check("filtered out" in meta, f"filter applied -> {meta[:80]}")
        pg.select_option("#ctlType", "heatmap")
        pg.wait_for_timeout(1200)
        check(pg.locator("#exploreChart svg rect.mark").count() > 4, "heatmap rendered")
        pg.screenshot(path=SHOTS / "5-explore.png", full_page=True)

        print("== data tab ==")
        pg.locator('.tab[data-tab="data"]').click()
        pg.wait_for_selector("#rowTable table tbody tr", timeout=20000)
        check(pg.locator("#schemaTable tbody tr").count() >= 10,
              f"schema rows: {pg.locator('#schemaTable tbody tr').count()}")
        check(pg.locator("#rowTable tbody tr").count() > 10, "row table populated")
        pg.locator("#rowSearch").fill("Mumbai")
        pg.wait_for_timeout(900)
        rc = pg.locator("#rowCount").inner_text()
        check("matching rows" in rc, f"search works -> {rc}")
        pg.screenshot(path=SHOTS / "6-data.png", full_page=True)

        print("== settings modal ==")
        pg.locator("#btnSettings").click()
        pg.wait_for_selector(".modal", timeout=5000)
        check(pg.locator("#mBase").is_visible(), "settings modal opens")
        pg.locator("#mCancel").click()
        pg.wait_for_timeout(300)
        check(pg.locator(".modal").count() == 0, "settings modal closes")

        print("== upload a file ==")
        pg.locator("#btnReset").click()
        pg.wait_for_timeout(400)
        csv = "id,team,score\n" + "\n".join(f"{i},{'alpha' if i % 2 else 'beta'},{i * 7 % 53}" for i in range(60))
        f = SHOTS / "upload.csv"
        f.write_text(csv)
        pg.set_input_files("#fileInput", str(f))
        pg.wait_for_selector("#kpiRow .kpi", timeout=30000)
        pg.wait_for_timeout(1000)
        check("upload.csv" in pg.locator("#dsName").inner_text(), "uploaded file loaded")
        check(pg.locator("#insightList .insight").count() >= 1, "insights generated for the upload")
        pg.screenshot(path=SHOTS / "7-upload.png", full_page=True)

        b.close()

    print("\n== console/page errors ==")
    real = [e for e in errors if "favicon" not in e]
    for e in real[:15]:
        print("  " + e)
    if not real:
        print("  none")
    failed = [c for c in checks if not c[0]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed or real:
        print("FAILURES:", [c[1] for c in failed])
        sys.exit(1)


if __name__ == "__main__":
    main()
