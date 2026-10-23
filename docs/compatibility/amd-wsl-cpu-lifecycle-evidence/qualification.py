"""One owned nonrecursive WSL CPU target/sentinel pair; no inference."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
P = ROOT.parent / 'artifacts/amd-wsl-cpu-lifecycle-8f175de'
OUT = P / 'cpu-once'
BASE = '8f175de809f3be304cfc92179cb42905a01b8f12'
BRANCH = 'codex/amd-wsl-cpu-lifecycle'
FIXTURE = E / 'fixture.py'
REL = 'docs/compatibility/amd-wsl-cpu-lifecycle-evidence/'
PINS = ('api/services/amd_stage_process.py', 'api/services/amd_stage_bootstrap.py',
        'api/services/amd_completed_result.py', 'api/services/amd_upstream_job.py',
        'docs/compatibility/amd-windows-lifecycle-outside-evidence/windows_qualification.py',
        'api/services/amd_windows_authority.py',
        'api/tests/test_amd_windows_lifecycle_authority.py', 'api/tests/test_amd_windows_authority.py',
        'api/services/amd_wsl_session.py',
        *(REL + name for name in ('qualification.py','coordinator.py','fixture.py','test_fixture.py','test_qualification.py','outside_command.ps1')))
# Reuse the qualified publisher only; never call the historical host or worker.
_spec = importlib.util.spec_from_file_location('historical_publisher', ROOT / PINS[4])
_publisher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_publisher)
save, read, sha = _publisher.save, _publisher.read, _publisher.sha
sys.path.insert(0, str(ROOT / 'api'))
from services.amd_stage_process import OwnedStageRunner, StageCommand, _OwnedJob, _host_health, _error_detail
from services.amd_completed_result import CompletionContext, _plain


def source_pins():
    return {name: sha(ROOT / name) for name in PINS}


def registration_identity():
    import winreg
    found = []
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Lxss') as root:
        for index in range(winreg.QueryInfoKey(root)[0]):
            with winreg.OpenKey(root,winreg.EnumKey(root,index)) as key:
                if winreg.QueryValueEx(key,'DistributionName')[0] == 'Modly-AMD-Ubuntu-24.04':
                    found.append({name:winreg.QueryValueEx(key,name)[0] for name in
                        ('DistributionName','DefaultUid','Version','BasePath')})
    if len(found) != 1 or found[0]['Version'] != 2:
        raise ValueError('Exact named WSL2 registration required')
    return found[0]


def wsl_unchanged(admission):
    if (sha(P/'registration-before-private.json') != admission['registration_before_sha256']
            or registration_identity() != read(P/'registration-before-private.json')['registration']
            or sha(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe') != admission['wsl_executable_sha256']):
        raise ValueError('Original WSL registration/executable changed')


def preflight():
    admission = read(E / 'admission.json')
    if sys.platform != 'win32' or Path(sys.executable).resolve() != Path(admission['python_executable']).resolve():
        raise ValueError('Original Windows API interpreter required')
    if sha(sys.executable) != admission['python_executable_sha256']:
        raise ValueError('Interpreter changed')
    wsl_unchanged(admission)
    ref = 'refs/heads/' + BRANCH
    if (ROOT / '.git/HEAD').read_text().strip() != 'ref: ' + ref or (ROOT / '.git' / ref).read_text().strip() != BASE:
        raise ValueError('Exact reviewed branch/base required')
    pins = source_pins()
    if admission['base'] != BASE or admission['branch'] != BRANCH or admission['source_pins'] != pins:
        raise ValueError('Exact reviewed source identity required')
    _plain(P); _plain(OUT)
    if not P.is_dir() or OUT.exists():
        raise ValueError('Fresh exclusive attempt required')
    review = read(E / 'preexecution-review.json')
    if review.get('verdict') != 'PASS' or review.get('source_pins') != pins:
        raise ValueError('Independent exact-source review required')
    context_hash = sha(P / 'expected-context.json')
    if admission.get('expected_context_sha256') != context_hash or review.get('expected_context_sha256') != context_hash:
        raise ValueError('Reviewed outside-user identity changed')
    return admission, pins


def worker():
    spec=importlib.util.spec_from_file_location('authority_coordinator',E/'coordinator.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.run(sys.modules[__name__])


def host():
    began = time.monotonic(); admission, pins = preflight()
    expected_context = read(P / 'expected-context.json')
    actual_context = read(P / 'outside-context.json')
    if (any(actual_context.get(k) != expected_context[k] for k in ('sid', 'name'))
            or any(actual_context.get(k) is not False for k in ('administrator_role_enabled', 'is_system', 'is_anonymous'))
            or actual_context.get('requested_tool_override') != 'require_escalated'):
        raise ValueError('Reviewed non-elevated outside-context wrapper required')
    initial = _host_health(P)
    OUT.mkdir(exist_ok=False)
    attempt = dict(base=BASE, branch=BRANCH, source_pins=pins, interpreter_sha256=sha(sys.executable),
        initial_resources=initial, admission_sha256=sha(E / 'admission.json'),
        review_sha256=sha(E / 'preexecution-review.json'), maximum_control_pairs=1, maximum_stage_invocations=2,
        memory_limit=6*1024**3, ram_floor=12*1024**3, disk_floor=2*1024**3, stage_seconds=25,
        active_end=began+70, grace_end=began+82, force_end=began+87, hard_end=began+90)
    save(OUT / 'attempt.json', attempt)
    finished = threading.Event()
    def watchdog():
        if not finished.wait(max(0, attempt['hard_end']-time.monotonic())): os._exit(124)
    threading.Thread(target=watchdog, daemon=True).start()
    result = dict(classification='BLOCKED', samples=[], errors=[], cleanup_errors=[],
                  forced_outer_cleanup=False, windows=None, initial_resources=initial)
    job = child = None; attached = gate_open = False; phase = 'outer.create'
    def snapshot():
        job.phase = phase; result['windows'] = job.snapshot(); return result['windows']
    def sample():
        result['samples'].append(dict(monotonic_s=time.monotonic(), **_host_health(P)))
    try:
        job = _OwnedJob()
        with (OUT/'worker.stdout.txt').open('xb') as stdout, (OUT/'worker.stderr.txt').open('xb') as stderr:
            phase = 'outer.Popen'
            child = subprocess.Popen([sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve()), '--worker'],
                stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, cwd=str(P),
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES=''),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            phase='outer.attach'; job.phase=phase; job.attach(child); attached=True
            phase='outer.runtime-ready'
            while not (OUT/'worker-ready.json').exists():
                if child.poll() is not None or time.monotonic() >= attempt['active_end']:
                    raise RuntimeError('Coordinator runtime readiness missing')
                sample(); time.sleep(.1)
            ready=read(OUT/'worker-ready.json'); observed=snapshot()
            rows=[r for r in observed['process_identities'] if r['pid']==ready['pid'] and r.get('job_membership_verified') is True and r.get('observed_alive', {}).get('alive') is True]
            if len(rows)!=1 or ready['pid'] not in observed['enumerated_pids']:
                raise ValueError('Coordinator not natively authenticated before GO')
            result['before_GO']=dict(ready=ready, windows=observed)
            wsl_unchanged(admission)
            phase='outer.GO'; child.stdin.write(b'GO\n'); child.stdin.close(); gate_open=True
            while child.poll() is None:
                phase='outer.observe'
                if time.monotonic() >= attempt['active_end']: raise TimeoutError('Outer active deadline')
                sample(); snapshot(); time.sleep(.25)
            phase='outer.natural-settle'; end=min(time.monotonic()+1, attempt['grace_end'])
            while snapshot()['active'] and time.monotonic()<end: time.sleep(.025)
            if snapshot()['active'] or child.returncode != 0 or read(OUT/'pair-result.json')['classification']!='PASS':
                raise RuntimeError('Invocation or natural outer settlement failed')
            result['classification']='PASS'
    except BaseException as error:
        result['errors'].append(_error_detail(error, phase))
    finally:
        try:
            if child is not None and child.stdin is not None and not child.stdin.closed: child.stdin.close()
            if child is not None and child.poll() is None and gate_open:
                phase='outer.cleanup-cooperative'
                save(OUT/'host-cancel.json', dict(action='CANCEL', attempt_sha256=sha(OUT/'attempt.json')))
                end=min(time.monotonic()+12, attempt['grace_end'])
                while child.poll() is None and time.monotonic()<end: sample(); time.sleep(.1)
        except BaseException as error: result['cleanup_errors'].append(_error_detail(error, phase))
        if job is not None:
            active=True
            try: phase='outer.final-accounting'; active=snapshot()['active']!=0
            except BaseException as error: result['cleanup_errors'].append(_error_detail(error, phase))
            if active:
                result['forced_outer_cleanup']=True
                try:
                    phase='outer.force-containment'; job.phase=phase; job.terminate()
                    end=min(time.monotonic()+4, attempt['force_end'])
                    while snapshot()['active'] and time.monotonic()<end: time.sleep(.025)
                except BaseException as error: result['cleanup_errors'].append(_error_detail(error, phase))
            try: phase='outer.close'; snapshot(); job.phase=phase; job.close()
            except BaseException as error: result['cleanup_errors'].append(_error_detail(error, phase))
        try:
            if child is not None:
                if not attached and child.poll() is None: child.kill(); result['forced_outer_cleanup']=True
                child.wait(timeout=max(.001, min(2, attempt['hard_end']-time.monotonic()-1)))
                result['returncode']=child.returncode
            result['sources_unchanged']=source_pins()==pins
            result['admission_unchanged']=sha(E/'admission.json')==attempt['admission_sha256']
            result['review_unchanged']=sha(E/'preexecution-review.json')==attempt['review_sha256']
            result['interpreter_unchanged']=sha(sys.executable)==attempt['interpreter_sha256']
            wsl_unchanged(admission); result['wsl_registration_and_executable_unchanged']=True
            sample()
        except BaseException as error: result['cleanup_errors'].append(_error_detail(error, 'outer.final-verification'))
        result['elapsed_s']=time.monotonic()-began
        if (result['errors'] or result['cleanup_errors'] or result['forced_outer_cleanup']
                or result['elapsed_s']>=90 or result.get('returncode')!=0
                or not isinstance(result['windows'], dict) or result['windows'].get('active')!=0
                or not all(result.get(k) is True for k in ('sources_unchanged','admission_unchanged','review_unchanged','interpreter_unchanged','wsl_registration_and_executable_unchanged'))):
            result['classification']='BLOCKED'
        save(OUT/'host-result.json', result); finished.set()
    print(json.dumps(dict(classification=result['classification'],elapsed_s=result['elapsed_s'])))
    return 0 if result['classification']=='PASS' else 1


def entry(mode):
    try:
        if mode == ['--preflight']:
            preflight(); print('STATIC_PREFLIGHT_PASS'); return 0
        if mode == ['--worker']: return worker()
        if mode == []: return host()
        raise ValueError('Unsupported invocation')
    except BaseException as error:
        # Independent location, exclusive write, complete original error; no
        # dependence on hardlink publication and never replaces original exception.
        row=_error_detail(error, 'qualification.entry')
        publication=getattr(error,'modly_publication',{})
        row['publication']=publication
        row['operation_paths']={key:_publisher._path_snapshot(publication.get(key)) for key in ('staging','final')}
        try:
            directory=P/'diagnostics'; directory.mkdir(exist_ok=True)
            with (directory/(uuid.uuid4().hex+'.json')).open('x',encoding='utf-8') as stream:
                json.dump(row,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            print(json.dumps(row),file=sys.stderr,flush=True)
        raise


if __name__=='__main__': raise SystemExit(entry(sys.argv[1:]))
