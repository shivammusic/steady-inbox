"""Download model data from a pinned official revision and verify local hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'model-manifest.json').read_text())
model_dir = root / 'models'
model_dir.mkdir(exist_ok=True)
for item in manifest['files']:
    target = model_dir / item['local_name']
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
        url = f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}/{item['remote_path']}"
        print(f"Downloading {item['remote_path']} from the pinned official model revision...")
        with urllib.request.urlopen(url, timeout=90) as response, target.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
    if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError(f'Checksum mismatch for {target.name}; do not run this model.')
    print(f'Verified {target.name}')
print('Model ready. Future inference uses local files and needs no network.')
