"""Package only the multimodal changes; never overwrite server RGB recipe files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--out",required=True)
a = p.parse_args()
files = [p for p in (ROOT/"mm_yolo").rglob("*") if p.is_file() and p.suffix in (".py",".md") and "__pycache__" not in p.parts]
files += [ROOT/p for p in ("train_multimodal.py","model_catalog.py","configs/mm_v3_server.json","MULTIMODAL_V3.md")]
files += [p for p in (ROOT/"ops").glob("*.py")]
manifest = {str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(a.out,"x",zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path,str(path.relative_to(ROOT)).replace("\\","/"))
    z.writestr("v3_manifest.json",json.dumps(manifest,indent=2))
print(json.dumps({"bundle":a.out,"files":len(files),"sha256":hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}))
