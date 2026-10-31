"""First-party adapter. Reused Blender entry owns the sole GO/EOF read."""
import sys,runpy
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import amd_static_prop_recovery as recovery
stage=sys.argv[sys.argv.index('--')+1]
if stage not in ('export',*recovery.RENDERS):raise ValueError('Declared CPU Blender stage only')
recovery.bind(stage)
runpy.run_path(str(Path(__file__).with_name('amd_static_prop_blender.py')),run_name='__main__')
