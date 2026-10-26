"""Admission-only recorder. Even PASS never creates an environment or runs operators."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
path=Path(__file__).with_name('amd_zombie_d_handoff.py')
spec=importlib.util.spec_from_file_location('_pixal_original_windows_gate',path)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


def collect(read,sleep):
    result=dict(classification='NO_GO_WINDOWS_ADMISSION',required_available_bytes=22*1024**3,
                preferred_available_bytes=24*1024**3,preferred_is_mandatory=False,
                requested_readings=3,spacing_seconds=4,readings=[],WSL_launched=False,
                GPU_initialized=False,environment_created=False,package_or_model_downloaded=False,
                native_operations_executed=False)
    try:
        original.fresh_admission(read,sleep,result['readings'].append)
        result['classification']='PASS_WINDOWS_ADMISSION'
    except Exception as exc:
        result['stop_reason']=type(exc).__name__+': '+str(exc)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Only take original Windows admission readings')
    args=parser.parse_args(argv)
    if not args.check:
        print('DRY_RUN: no live readings, WSL, environment creation, package download or GPU operation')
        return 0
    artifacts=original.plain(ROOT.parent/'artifacts')
    base=artifacts/('amd-pixal3d-admission-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex)
    base.mkdir(exist_ok=False)
    result=collect(original.windows_memory,time.sleep)
    result.update(measurement='Original qualified GlobalMemoryStatusEx/GetProcessMemoryInfo',
                  source_gate_sha256=original.digest(path),
                  source_reader_sha256=original.digest(path.with_name('amd_zombie_ram_diagnostic.py')),
                  free_output_disk_bytes=shutil.disk_usage(base).free,
                  free_D_disk_bytes=shutil.disk_usage('D:/').free,
                  windows_path_unused=not Path('D:/AI/Modly-Pixal3D-Proof-v1').exists(),
                  no_guest_path_or_GPU_assertion=True,
                  admission_only_not_compatibility_qualification=True)
    if result['free_output_disk_bytes']<2*1024**3 or not result['windows_path_unused']:
        result.update(classification='NO_GO_WINDOWS_ADMISSION',stop_reason='Original disk floor or exclusive Windows path failed')
    original.save(base/'windows-admission.json',result)
    print(str(base/'windows-admission.json'))
    print(result['classification']+'; no automatic continuation')
    return 0 if result['classification']=='PASS_WINDOWS_ADMISSION' else 2


if __name__=='__main__':raise SystemExit(main())
