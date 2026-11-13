"""Extract the visible text of a built report page (what a reader sees, incl. figure captions and tables)."""
import sys, asyncio
from playwright.async_api import async_playwright
async def main(src, out):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        html = open(src, encoding="utf8").read()
        await pg.set_content("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>" + html, wait_until="networkidle")
        await pg.wait_for_timeout(1500)
        if len(sys.argv) > 3: await pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
        t = await pg.evaluate("document.body.innerText")
        open(out, "w", encoding="utf8").write(t)
        print("chars", len(t), "words", len(t.split()), "errors", errs[:5])
        await b.close()
asyncio.run(main(sys.argv[1], sys.argv[2]))
