import hashlib
import json
from pathlib import Path

import torch

src = Path('/root/autodl-tmp/runs/v521_geometry_warmup_spatial_exact_20260925_v2/best.pt')
dst = src.with_name('best_stagea_init.pt')
checkpoint = torch.load(src, map_location='cpu', weights_only=False)
meta = dict(checkpoint.get('meta') or {})
meta.update({
    'modalities': ['rgb', 'ir', 'dep'],
    'canvas': [736, 1280],
    'imgsz': [736, 1280],
    'name': 'v521_geometry_warmup_stagea_init',
})
checkpoint['meta'] = meta
torch.save(checkpoint, dst)
print(json.dumps({
    'source': str(src),
    'destination': str(dst),
    'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
    'destination_sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
    'meta': meta,
}, indent=2))
