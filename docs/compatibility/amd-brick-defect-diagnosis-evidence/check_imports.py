"""CPU-only regression for the real guarded numerical import closure."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose
sys.addaudithook(diagnose.runtime_guard)
import numpy
import scipy
from scipy import ndimage
from PIL import Image
import topology
import projection
diagnose.runtime_guard('socket.gethostname', ())
rejected = []
for event, args in [('socket.connect', ()), ('socket.getaddrinfo', ()),
                    ('subprocess.Popen', ()), ('import', ('torch',)),
                    ('open', (str(diagnose.P.parent / 'foreign.txt'), 'w', 0))]:
    try:
        diagnose.runtime_guard(event, args)
    except (RuntimeError, ValueError):
        rejected.append(event)
    else:
        raise AssertionError('Unexpected permission: ' + event)
print('PASS: guarded installed NumPy/SciPy ndimage/PIL and local modules import; '
      'local hostname query allowed, five GPU/network/process/foreign-write negatives rejected.', flush=True)
