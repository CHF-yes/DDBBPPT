"""Non-invasive model registry: existing server RGB settings remain authoritative."""
from pathlib import Path
import models_config

if "rgb_hq_11m" not in models_config.get_keys():
    raise RuntimeError("the authoritative server RGB YOLO11m recipe is missing")
MODELS = {"rgb_hq_11m":{"entrypoint":"train_rgb.py","config":"rgb_hq_11m","kind":"rgb"}}
MODELS["mm_v3_11m_p2_rect"] = {"entrypoint":"train_multimodal.py","config":"configs/mm_v3_server.json","kind":"multimodal"}

if __name__ == "__main__":
    import json
    print(json.dumps(MODELS,ensure_ascii=False,indent=2))
