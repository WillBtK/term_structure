"""Inject viz/data.json into the artifact template -> viz/artifact.html."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
tpl = (HERE / "artifact_template.html").read_text()
data = (HERE / "data.json").read_text()
assert "__DATA_BLOB__" in tpl, "template missing __DATA_BLOB__ placeholder"
# guard against accidental </script> inside the JSON breaking the data-blob tag
data = data.replace("</", "<\\/")
out = tpl.replace("__DATA_BLOB__", data)
(HERE / "artifact.html").write_text(out)
print("wrote", HERE / "artifact.html", f"{len(out)/1024:.0f} KB")
