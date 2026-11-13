"""Assemble the page from template.html + data.json + prose.json.

Usage: python report/assemble.py analysis/data.json report/prose.json report/report.html
"""
import json, sys
from pathlib import Path
T = (Path(__file__).resolve().parent / "template.html").read_text()
data = Path(sys.argv[1]).read_text()
prose = Path(sys.argv[2]).read_text()
json.loads(data); json.loads(prose)
safe = lambda s: s.replace("</", "<\\/").replace("<!--", "<\\!--")
out = T.replace("__DATA__", safe(data)).replace("__PROSE__", safe(prose))
Path(sys.argv[3]).write_text(out)
print("wrote", sys.argv[3], round(len(out) / 1e6, 2), "MB")
