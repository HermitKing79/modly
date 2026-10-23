"""One test-only native Windows pair through the UNCHANGED production runner.

Importing this module starts nothing. --preflight never reserves an attempt.
Raw identity records stay private. This module issues no UpstreamJob authority.
"""
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
P = ROOT.parent / 'artifacts/amd-minimal-windows-supervisor-df30cb3'
OUT = P / 'cpu-once'
BASE = 'df30cb32e87ab528d9c9961ea0953ec4798f6529'
BRANCH = 'codex/amd-minimal-windows-supervisor'
FIXTURE = E / 'fixture.py'
REL = 'docs/compatibility/amd-minimal-windows-supervisor-evidence/'
PINS = ('api/services/amd_stage_process.py', 'api/services/amd_stage_bootstrap.py',
        'api/services/amd_completed_result.py', 'api/services/amd_upstream_job.py',
        'docs/compatibility/amd-windows-lifecycle-outside-evidence/windows_qualification.py',
        *(REL + name for name in ('qualification.py', 'fixture.py', 'test_fixture.py',
                                  'test_qualification.py', 'outside_command.ps1')))
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


def live_proof(ready, snapshot, expected, now, after=0):
    for key, value in expected.items():
        if ready.get(key) != value:
            raise ValueError('Foreign fixture readiness: ' + key)
    pid = ready.get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('Actual fixture PID required')
    if not isinstance(snapshot, dict):
        return None
    stamp = snapshot.get('monotonic_s', -1)
    if not after <= stamp <= now or now - stamp > 1 or pid not in snapshot.get('enumerated_pids', []):
        return None
    identities = []
    for row in snapshot.get('process_identities', []):
        native = row.get('image_authentication', {})
        exit_row = row.get('exit_observation', {})
        if (row.get('pid') == pid and row.get('job_membership_verified') is True
                and type(row.get('creation_filetime')) is int and row['creation_filetime'] > 0
                and row.get('job_owner') == snapshot.get('job_owner') and row.get('job_owner')
                and row.get('job_handle') and row.get('observed_alive', {}).get('alive') is True
                and row.get('identity_source') == 'native-image-query'
                and native.get('api') == 'QueryFullProcessImageNameW' and native.get('succeeded') is True
                and native.get('observed_utc') and row.get('image')
                and all(native.get(key) == row.get(key) for key in ('pid', 'creation_filetime', 'job_owner', 'job_handle', 'image'))
                and exit_row.get('api') == 'GetExitCodeProcess' and exit_row.get('succeeded') is True
                and exit_row.get('exit_code') == 259):
            identities.append(row)
    if len(identities) != 1:
        return None
    return dict(ready=ready, observation=snapshot, identity=identities[0])


def proof_after_fence(ready, snapshot, expected, now, fence):
    if not isinstance(snapshot, dict) or snapshot.get('monotonic_s', -1) <= fence:
        return None
    return live_proof(ready, snapshot, expected, now, fence)


def settled(row, terminal):
    return (isinstance(row, dict) and row.get('terminal') == terminal
        and row.get('natural') is True and row.get('forced') is False
        and row.get('windows_settled') is True and row.get('windows', {}).get('active') == 0
        and row.get('returncode') == row.get('bootstrap_returncode') == 0
        and 0 < row.get('elapsed_s', 999) < 25 and not row.get('diagnostic_errors')
        and not any(key.endswith('_error') for key in row))


def preflight():
    admission = read(E / 'admission.json')
    if sys.platform != 'win32' or Path(sys.executable).resolve() != Path(admission['python_executable']).resolve():
        raise ValueError('Original Windows API interpreter required')
    if sha(sys.executable) != admission['python_executable_sha256']:
        raise ValueError('Interpreter changed')
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
    save(OUT / 'worker-ready.json', dict(pid=os.getpid(), monotonic_s=time.monotonic()))
    if sys.stdin.buffer.readline(32) != b'GO\n':
        raise ValueError('Outer owned Job GO required')
    attempt = read(OUT / 'attempt.json')
    if attempt['source_pins'] != source_pins() or attempt['interpreter_sha256'] != sha(sys.executable):
        raise ValueError('Worker source identity changed')
    states, commands, runners, threads, expected = {}, {}, {}, {}, {}
    events = {name: threading.Event() for name in ('target', 'sentinel')}
    done = {name: threading.Event() for name in events}
    pair = dict(classification='BLOCKED', synthetic_nonrecursive_only=True, parent_stage_pairs=1,
                timeline=[], pre_cancel=None, post_target=None, runner_states=states)

    def point(event, **fields):
        pair['timeline'].append(dict(event=event, monotonic_s=time.monotonic(), **fields))

    def check():
        marker = OUT / 'host-cancel.json'
        if marker.exists():
            if read(marker) != dict(action='CANCEL', attempt_sha256=sha(OUT / 'attempt.json')):
                raise ValueError('Foreign outer cancellation')
            raise InterruptedError('Outer host cancelled')
        if time.monotonic() >= attempt['active_end']:
            raise TimeoutError('Outer active deadline')

    def run(name):
        try:
            point(name + '-runner-entry')
            receipt = runners[name].run(commands[name], events[name], tick=check)
            states[name] = dict(receipt=receipt, finished_monotonic_s=time.monotonic())
        except BaseException as error:
            states[name] = dict(exception=_error_detail(error, 'coordinator.' + name))
        finally:
            done[name].set()

    def live(name, after=0, fence=None):
        path = commands[name].cwd / 'ready.json'
        if not path.exists(): return None
        if fence is not None:
            return proof_after_fence(read(path), runners[name].last_windows_observation,
                                     expected[name], time.monotonic(), fence)
        return live_proof(read(path), runners[name].last_windows_observation,
                          expected[name], time.monotonic(), after)

    try:
        for name in events:
            context = CompletionContext.new('synthetic-' + name, b'CPU-fixture-only-' + name.encode(),
                {'purpose': 'nonrecursive-supervisor-qualification'}, mode='replay_fixture',
                model_id='test-only/no-inference').to_dict()
            leaf = OUT / (name + '-fixture'); leaf.mkdir()
            expected[name] = dict(context=context, stage_id='minimal-' + name, role=name,
                                  release_nonce=uuid.uuid4().hex, job_nonce=uuid.uuid4().hex)
            value = expected[name]
            commands[name] = StageCommand(value['stage_id'], 'windows',
                (sys.executable, '-B', '-I', '-S', str(FIXTURE), '--role', name,
                 '--private-leaf', str(leaf), '--release-nonce', value['release_nonce'],
                 '--job-nonce', value['job_nonce']), leaf, context, 25)
            runners[name] = OwnedStageRunner(OUT / (name + '-runner'))
            threads[name] = threading.Thread(target=run, args=(name,), daemon=True)
        pair['expected'] = expected
        point('pair-launch')
        for thread in threads.values(): thread.start()
        # No descendant-readiness subdeadline: unchanged runner25s governs both.
        while True:
            check()
            if any(flag.is_set() for flag in done.values()):
                raise RuntimeError('Runner exited before both authenticated ready')
            a, b = live('target'), live('sentinel')
            if a and b:
                if (a['identity']['job_owner'] == b['identity']['job_owner']
                        or a['identity']['job_handle'] == b['identity']['job_handle']
                        or a['identity']['pid'] == b['identity']['pid']):
                    raise RuntimeError('Independent private Jobs/processes required')
                pair['pre_cancel'] = dict(target=a, sentinel=b); break
            time.sleep(.025)
        point('both-authenticated-ready')
        pair['cancel_at'] = time.monotonic(); events['target'].set()
        point('target-only-cancel')
        while not done['target'].wait(.025):
            check()
            if done['sentinel'].is_set() or events['sentinel'].is_set():
                raise RuntimeError('Sentinel ended during target cancellation')
        if not settled(states['target'].get('receipt'), 'cancelled'):
            raise RuntimeError('Target did not settle naturally with acknowledgement')
        finished = states['target']['finished_monotonic_s']
        ack = read(OUT / 'target-runner/minimal-target/cancel-ack.json')
        if any(ack.get(k) != v for k, v in expected['target'].items()) or ack.get('terminal') != 'cancelled' or ack.get('nonce') != expected['target']['context']['nonce'] or ack.get('pid') != pair['pre_cancel']['target']['identity']['pid']:
            raise ValueError('Target exact cooperative acknowledgement differs')
        pair['target_ack'] = ack
        point('target-natural-settlement', finished_monotonic_s=finished)
        fence = None
        while True:
            check()
            if done['sentinel'].is_set() or events['sentinel'].is_set():
                raise RuntimeError('Sentinel failed to survive target settlement')
            b = live('sentinel', finished, fence)
            if b:
                original = pair['pre_cancel']['sentinel']['identity']
                if any(b['identity'][k] != original[k] for k in ('pid', 'creation_filetime', 'job_owner', 'job_handle', 'image')):
                    raise RuntimeError('Sentinel identity changed')
                if fence is None:
                    # Runner stamps snapshots after their native queries. Its next
                    # sequential snapshot must START after this completed fence.
                    pair['post_target_fence'] = b
                    fence = b['observation']['monotonic_s']
                else:
                    pair['post_target'] = b; break
            time.sleep(.025)
        point('sentinel-survived')
        release = dict(action='RELEASE', context=expected['sentinel']['context'],
            stage_id=expected['sentinel']['stage_id'], pid=b['identity']['pid'],
            release_nonce=expected['sentinel']['release_nonce'], job_nonce=expected['sentinel']['job_nonce'])
        save(commands['sentinel'].cwd / 'release.json', release)
        point('sentinel-release-published')
        while not done['sentinel'].wait(.025): check()
        if not settled(states['sentinel'].get('receipt'), 'succeeded') or events['sentinel'].is_set():
            raise RuntimeError('Sentinel natural success failed')
        released = read(commands['sentinel'].cwd / 'released.json')
        if (any(released.get(k) != v for k, v in expected['sentinel'].items())
                or released.get('pid') != b['identity']['pid']
                or released.get('terminal') != 'succeeded' or released.get('action') != 'RELEASE'
                or released.get('nonce') != expected['sentinel']['context']['nonce']):
            raise ValueError('Sentinel release receipt differs')
        pair['sentinel_released'] = released
        if any(n.split('.')[0] in ('torch', 'triton', 'flex_gemm') for n in sys.modules):
            raise RuntimeError('Forbidden model runtime')
        check(); pair['classification'] = 'PASS'; point('pair-natural-completion')
    except BaseException as error:
        pair['error'] = _error_detail(error, 'coordinator.controls')
    finally:
        if pair['classification'] != 'PASS':
            for flag in events.values(): flag.set()
        end = min(time.monotonic() + 12, attempt['grace_end'])
        for thread in threads.values():
            if thread.ident is not None: thread.join(max(0, end - time.monotonic()))
        pair['threads_alive'] = [name for name, thread in threads.items() if thread.is_alive()]
        if pair['threads_alive']: pair['classification'] = 'BLOCKED'
        save(OUT / 'pair-result.json', pair)
    return 0 if pair['classification'] == 'PASS' else 1


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
        review_sha256=sha(E / 'preexecution-review.json'), maximum_pairs=1,
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
            phase='outer.GO'; child.stdin.write(b'GO\n'); child.stdin.close(); gate_open=True
            while child.poll() is None:
                phase='outer.observe'
                if time.monotonic() >= attempt['active_end']: raise TimeoutError('Outer active deadline')
                sample(); snapshot(); time.sleep(.25)
            phase='outer.natural-settle'; end=min(time.monotonic()+1, attempt['grace_end'])
            while snapshot()['active'] and time.monotonic()<end: time.sleep(.025)
            if snapshot()['active'] or child.returncode != 0 or read(OUT/'pair-result.json')['classification']!='PASS':
                raise RuntimeError('Pair or natural outer settlement failed')
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
            sample()
        except BaseException as error: result['cleanup_errors'].append(_error_detail(error, 'outer.final-verification'))
        result['elapsed_s']=time.monotonic()-began
        if (result['errors'] or result['cleanup_errors'] or result['forced_outer_cleanup']
                or result['elapsed_s']>=90 or result.get('returncode')!=0
                or not isinstance(result['windows'], dict) or result['windows'].get('active')!=0
                or not all(result.get(k) is True for k in ('sources_unchanged','admission_unchanged','review_unchanged','interpreter_unchanged'))):
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
