"""Detach the authorized server training from SSH; fail on an existing active run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--root",required=True)
p.add_argument("--labels",required=True)
p.add_argument("--weights",required=True)
p.add_argument("--out",required=True)
p.add_argument("--resume",action="store_true")
a = p.parse_args()
cfg = json.loads((ROOT/"configs/mm_v3_server.json").read_text(encoding="utf-8"))
run = Path(a.out).resolve()/cfg["name"]
status = run/"pipeline_status.json"
if status.exists():
    previous = json.loads(status.read_text(encoding="utf-8"))
    pid = previous.get("pid",0)
    if pid and Path(f"/proc/{pid}/cmdline").exists():
        raise RuntimeError(f"existing pipeline PID {pid}; refusing duplicate")
    if not a.resume:
        raise FileExistsError(f"existing pipeline at {run}; explicit resume required")
run.mkdir(parents=True,exist_ok=True)
cmd = [sys.executable,"-u",str(ROOT/"train_multimodal.py"),"--root",a.root,"--labels",a.labels,
       "--weights",a.weights,"--out",a.out]
if a.resume:
    cmd += ["--resume"]
env = dict(os.environ,PYTHONUTF8="1",PYTHONIOENCODING="utf-8")
with (run/"launcher.log").open("a",encoding="utf-8") as log:
    process = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/"launcher.pid").write_text(str(process.pid),encoding="ascii")
print(json.dumps({"pid":process.pid,"run":str(run),"command":cmd},ensure_ascii=False))
