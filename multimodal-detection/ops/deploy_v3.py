"""Server-side, allowlisted overlay after recoverable code backup. No training data touched."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

p = argparse.ArgumentParser()
p.add_argument("--project",required=True)
p.add_argument("--bundle",required=True)
p.add_argument("--archive-root",required=True)
a = p.parse_args()
root = Path(a.project).resolve()
assert (root/"train_rgb.py").is_file() and (root/"mm_yolo/model.py").is_file()
protected = ["train_rgb.py","models_config.py","common/trainer.py","predict_rgb.py"]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {s:digest(root/s) for s in protected if (root/s).is_file()}
archive = Path(a.archive_root).resolve()/datetime.datetime.now().strftime("pre_v3_%Y%m%d_%H%M%S_%f")
archive.mkdir(parents=True,exist_ok=False)
def filt(info):
    if any(part in ("runs","weights","__pycache__","vendor",".git") for part in Path(info.name).parts):
        return None
    return info
with tarfile.open(archive/"code_before.tar.gz","w:gz") as tar:
    tar.add(root,arcname="multimodal-detection",filter=filt)
with zipfile.ZipFile(a.bundle) as z:
    manifest = json.loads(z.read("v3_manifest.json"))
    for name,expected in manifest.items():
        target = (root/name).resolve()
        target.relative_to(root)
        if name in protected or not (name.startswith(("mm_yolo/","ops/")) or name in ("train_multimodal.py","model_catalog.py","configs/mm_v3_server.json","MULTIMODAL_V3.md")):
            raise ValueError(f"forbidden overlay target {name}")
        content = z.read(name)
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError(f"corrupt bundle file {name}")
    for name,expected in manifest.items():
        target = root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(z.read(name))
retired = archive/"retired"
retired.mkdir()
for name in ("基线模型1","基线模型2","实验模型1","legacy","experiment1.py","MISSING_FILES.md","MODIFICATIONS.md"):
    source = (root/name).resolve()
    source.relative_to(root)
    if source.exists():
        shutil.move(str(source),str(retired/name))
after = {s:digest(root/s) for s in before}
assert before == after, "Protected RGB files changed!"
report = {"archive":str(archive),"project":str(root),"protected_rgb_sha256":after,"files":manifest}
(archive/"deployment.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
(root/"v3_deployment.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"archive":str(archive),"protected_rgb_unchanged":True,"files":len(manifest)},ensure_ascii=False))
