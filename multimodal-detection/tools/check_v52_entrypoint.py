"""Reproduce the sys.path of `python mm_yolo/train.py` in a fresh process."""
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
here = root / 'mm_yolo'
sys.path[:] = [str(here)] + [p for p in sys.path if p and Path(p).resolve() not in (root, here, root / 'vendor', root / 'tools')]
entry = sys.argv[1] if len(sys.argv) > 1 else 'train'
namespace = runpy.run_path(str(here / (entry + '.py')), run_name='v52_entrypoint_probe')
for name in ('data', 'model', 'config'):
    actual = Path(sys.modules[name].__file__).resolve()
    expected = here / (name + '.py')
    print(name, actual, 'correct=', actual == expected, flush=True)
    assert actual == expected, (actual, expected)
print('ENTRYPOINT_IMPORTS_PASSED', entry, flush=True)
