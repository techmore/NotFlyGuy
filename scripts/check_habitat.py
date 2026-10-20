"""Read-only runtime check: real JPEG frames and advancing simulation time."""
import base64
import io
import json
import time
from urllib.request import urlopen

from PIL import Image


def snapshot():
    with urlopen('http://127.0.0.1:8765/api/state', timeout=5) as response:
        return json.load(response)


first = snapshot()
assert first['frame'].startswith('data:image/jpeg;base64,')
with Image.open(io.BytesIO(base64.b64decode(first['frame'].split(',', 1)[1]))) as image:
    image.verify()
if first.get('paused'):
    print('PASS: paused habitat serves a valid JPEG')
else:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        second = snapshot()
        if second['timestep'] > first['timestep']:
            print('PASS: habitat simulation advances and serves a valid JPEG')
            break
        time.sleep(.1)
    else:
        raise AssertionError('Simulation did not advance within 10 seconds')
