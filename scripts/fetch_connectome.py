"""Download official MaleCNS v1.0 tables with size checks and provenance."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'malecns-v1.0'
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'body-annotations-male-cns-v1.0-minconf-0.5.feather': 14483314,
    'body-neurotransmitters-male-cns-v1.0.feather': 43282834,
    'connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather': 508025642,
}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        path = ROOT / name
        if not path.exists():
            part = path.with_suffix('.partial')
            with urlopen(BASE + name, timeout=60) as response, part.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if part.stat().st_size != expected:
                raise ValueError(f'Incomplete download: {name}')
            part.rename(path)
        if path.stat().st_size != expected:
            raise ValueError(f'Unexpected existing file size: {name}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        records.append({'file': name, 'url': BASE + name, 'bytes': expected, 'sha256': digest})
        print(f'Verified {name}: {expected} bytes', flush=True)
    manifest = {'dataset': 'MaleCNS v1.0', 'license': 'CC-BY',
                'source': 'https://male-cns.janelia.org/download/',
                'scope': 'traced-only connection weights, not the full segment graph',
                'files': records}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
