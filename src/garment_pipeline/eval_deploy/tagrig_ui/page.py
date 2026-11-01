"""The UI page (Korean, HTML + JS), served by GET /. The markup is page.html next to this file, read once as bytes, so the
served page is byte-identical to the string that used to be inlined in act_tagrig_ui.py."""
from pathlib import Path

PAGE = (Path(__file__).with_name("page.html")).read_bytes().decode("utf-8")
