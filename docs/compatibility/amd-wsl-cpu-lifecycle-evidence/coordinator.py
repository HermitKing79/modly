"""One WSL CPU pair, observed by the existing production runner. No authority."""
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor


def require(value, message):
    if not value:
        raise ValueError(message)


def records(path, context, stage):
    if not path.exists():
        return []
    data = path.read_bytes()
    require(len(data) <= 1024**2, 'Protocol size')
    # A live writer may have an unfinished final line; never interpret it.
    lines = data.splitlines(keepends=True)
    result = []
    for line in lines:
        if not line.endswith(b'\n'):
            break
        row = json.loads(line)
        require((row.get('context'), row.get('stage_id'), row.get('nonce')) ==
                (context, stage, context['nonce']), 'Foreign protocol binding')
        require(row.get('event') in ('control-received', 'ready', 'terminal', 'linux-client-exited'),
                'Unexpected protocol event')
        result.append(row)
    require(sum(r['event'] == 'ready' for r in result) <= 1, 'Repeated broker readiness')
    require(sum(r['event'] == 'terminal' for r in result) <= 1, 'Repeated broker terminal')
    return result


def linux_identity(ready):
    from services.amd_stage_process import _linux_ready
    require(_linux_ready(ready), 'Invalid broker Linux identity')
    b, l = ready['broker'], ready['leader']
    require(l.get('ppid') == b['pid'] and b['start_ticks'] <= l['start_ticks'], 'Broken broker ancestry')
    require(re.fullmatch(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}', ready.get('boot_id', '')),
            'Invalid Linux boot ID')
    m = ready['memory']
    require(m['aggregate_rss_limit_bytes'] == 6*1024**3 and m['atomic_cgroup_limit'] is False,
            'Linux memory policy mismatch')


def match_fixture(row, broker, command, mode):
    require((row.get('context'), row.get('stage_id'), row.get('nonce'), row.get('mode')) ==
            (command.context, command.stage_id, command.context['nonce'], mode), 'Foreign fixture readiness')
    require(row.get('uid') == row.get('euid') == 1000 and row.get('user') == 'modly', 'Wrong fixture user')
    require(row.get('rlimit_as') == [6*1024**3]*2, 'Inherited RLIMIT_AS mismatch')
    require(row.get('boot_id') == broker['boot_id'], 'Fixture boot ID mismatch')
    require(row.get('pid') == broker['leader']['pid'], 'Fixture leader mismatch')
    for key in ('pid', 'ppid', 'uid', 'start_ticks', 'pgrp', 'session'):
        require(row.get('identity', {}).get(key) == broker['leader'].get(key), 'Fixture identity mismatch: '+key)
    require(row.get('executable') == '/usr/bin/python3.12', 'Unexpected Linux interpreter')


def native_live(snapshot, now):
    require(isinstance(snapshot, dict) and 0 <= now-snapshot.get('monotonic_s', -99) < 1,
            'Fresh native Windows observation required')
    rows = snapshot.get('process_identities', [])
    live = []
    for row in rows:
        proof = row.get('image_authentication', {})
        if row.get('pid') not in snapshot.get('enumerated_pids', []):
            continue
        require(row.get('job_membership_verified') is True and row.get('job_owner') == snapshot.get('job_owner')
                and row.get('job_handle') and row.get('creation_filetime', 0) > 0
                and proof.get('api') == 'QueryFullProcessImageNameW' and proof.get('succeeded') is True
                and all(proof.get(k) == row.get(k) for k in ('pid','creation_filetime','image','job_owner','job_handle')),
                'Unauthenticated native Job identity')
        if row.get('observed_alive', {}).get('alive') is True:
            require(row.get('exit_observation', {}).get('api') == 'GetExitCodeProcess'
                    and row['exit_observation'].get('succeeded') is True
                    and row['exit_observation'].get('exit_code') == 259, 'Unverified Windows live status')
            live.append(row)
    expected = os.path.normcase(os.path.normpath(str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/wsl.exe')))
    require(any(os.path.normcase(os.path.normpath(x['image'])) == expected for x in live),
            'No live authenticated original WSL client')
    return live


def settled(row, terminal, command, ready):
    from services.amd_stage_process import _linux_settled
    require(row.get('terminal') == terminal and row.get('natural') is True and row.get('forced') is False
            and row.get('windows_settled') is True and row.get('linux_settled') is True
            and row.get('returncode') == row.get('bootstrap_returncode') == row.get('linux_client_returncode') == 0
            and row.get('elapsed_s', 999) < 25 and not row.get('diagnostic_errors')
            and not any(k.endswith('_error') for k in row), 'Stage not naturally settled')
    require(row.get('context') == command.context and row.get('nonce') == command.context['nonce']
            and row.get('stage_id') == command.stage_id, 'Receipt binding')
    w = row['windows']
    require(w['active'] == 0 and w['total_processes'] == len(w['process_identities']), 'Incomplete Windows accounting')
    require(all(x.get('job_membership_verified') is True for x in w['process_identities']), 'Windows membership')
    l = row.get('linux')
    require(_linux_settled(l) and l.get('terminal') == terminal and l.get('returncode') == 0
            and l.get('natural') is True and l.get('forced') is False and not l.get('errors'), 'Linux terminal missing/unsafe')
    require(all(l.get(k) == ready.get(k) for k in ('broker','leader','boot_id')), 'Linux identity changed')
    linux_identity(l)
    require(l['memory']['aggregate_rss_peak_bytes'] <= 6*1024**3, 'Linux RSS exceeded')
    require(len(l.get('reaped', [])) == 1 and l['reaped'][0]['pid'] == ready['leader']['pid']
            and l['reaped'][0]['status'] == 0, 'Nonrecursive leader reaping unverified')
    require(len(l.get('process_identities', [])) == 1, 'Unexpected Linux descendant population')
    original = ready['leader']
    require(all(l['process_identities'][0].get(k) == original[k]
                for k in ('pid','ppid','uid','start_ticks','session','pgrp')), 'Owned Linux process changed')
    controls = row.get('control_messages', [])
    cancels = [x for x in controls if x['action'] == 'CANCEL']
    require(all(x['flushed'] is True for x in controls), 'Unflushed control')
    if terminal == 'cancelled':
        require(len(cancels) == 1 and l.get('cooperative_cancelled') is True, 'Missing exact cooperative cancellation')
    else:
        require(not cancels and l.get('cooperative_cancelled') is False, 'Sentinel was cancelled')


def run(q):
    from services.amd_completed_result import CompletionContext
    from services.amd_stage_process import OwnedStageRunner, StageCommand, _guest_path, _error_detail
    out = q.OUT
    q.save(out/'worker-ready.json', dict(pid=os.getpid(), monotonic_s=time.monotonic()))
    if sys.stdin.buffer.readline(32) != b'GO\n':
        raise ValueError('Outer Job GO required')
    attempt = q.read(out/'attempt.json')
    require(attempt['source_pins'] == q.source_pins(), 'Changed reviewed source')
    result = dict(classification='BLOCKED', synthetic_only=True, native_control_pairs=1,
                  stage_invocations=2, WSL_UpstreamJob_authority='NOT QUALIFIED', neural='NOT RUN')
    began = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=2)
    events = [threading.Event(), threading.Event()]
    commands, runners, futures, admitted = [], [], [], []
    try:
        for index, mode in enumerate(('target', 'sentinel')):
            leaf = out/mode; leaf.mkdir()
            context = CompletionContext.new('wsl-cpu-'+mode, b'SYNTHETIC WSL CPU ONLY',
                        {'purpose':'one-nonrecursive-control-pair'}, model_id='synthetic/CPU-lifecycle').to_dict()
            command = StageCommand('wsl-'+mode, 'wsl',
                ('/usr/bin/python3','-B','-I','-S',_guest_path(q.FIXTURE),mode,_guest_path(leaf)),leaf,context,25)
            runner = OwnedStageRunner(out/(mode+'-process'))
            commands.append(command); runners.append(runner)
        def check():
            if (out/'host-cancel.json').exists():
                require(q.read(out/'host-cancel.json') == {'action':'CANCEL','attempt_sha256':q.sha(out/'attempt.json')},
                        'Foreign outer cancellation')
                raise InterruptedError('Outer cancellation')
            require(time.monotonic() < attempt['active_end'], 'Outer active deadline')
        for i in range(2):
            futures.append(pool.submit(runners[i].run, commands[i], events[i], check))
        while len(admitted) < 2:
            check()
            i = len(admitted); command = commands[i]
            path = command.cwd/'ready.json'
            if path.exists():
                rows = records(out/(('target','sentinel')[i]+'-process')/command.stage_id/'protocol.jsonl', command.context,command.stage_id)
                ready = [x for x in rows if x['event'] == 'ready']
                if ready:
                    linux_identity(ready[0]); fixture = q.read(path)
                    match_fixture(fixture,ready[0],command,('target','sentinel')[i])
                    native_live(runners[i].last_windows_observation,time.monotonic())
                    admitted.append(dict(broker_ready=ready[0],fixture_ready=fixture,
                                         windows=runners[i].last_windows_observation,host_monotonic_s=time.monotonic()))
                    continue
            require(not any(f.done() for f in futures), 'Stage ended before both authenticated readiness')
            time.sleep(.025)
        require(admitted[0]['broker_ready']['boot_id'] == admitted[1]['broker_ready']['boot_id'], 'Different Linux boots')
        require(admitted[0]['broker_ready']['leader']['session'] != admitted[1]['broker_ready']['leader']['session']
                and commands[0].context['nonce'] != commands[1].context['nonce'], 'Independent sessions/nonces required')
        result['admitted'] = admitted
        result['target_cancel_host_monotonic_s'] = time.monotonic(); events[0].set()
        target = futures[0].result(timeout=33)
        result['target'] = target
        settled(target,'cancelled',commands[0],admitted[0]['broker_ready'])
        ack = q.read(out/'target-process'/commands[0].stage_id/'cancel-ack.json')
        require(ack.get('terminal') == 'cancelled' and ack.get('context') == commands[0].context
                and ack.get('nonce') == commands[0].context['nonce'] and ack.get('stage_id') == commands[0].stage_id,
                'Original cooperative acknowledgement missing')
        result['target_ack_sha256'] = q.sha(out/'target-process'/commands[0].stage_id/'cancel-ack.json')
        require(not futures[1].done() and not events[1].is_set(), 'Sentinel failed independence')
        result['target_settled_host_monotonic_s'] = time.monotonic()
        challenge = dict(action='PROBE',context=commands[1].context,stage_id=commands[1].stage_id,
                         nonce=commands[1].context['nonce'],challenge=uuid.uuid4().hex)
        q.save(commands[1].cwd/'host-challenge.json',challenge)
        proof = commands[1].cwd/'alive-proof.json'
        while not proof.exists():
            check(); require(not futures[1].done(), 'Sentinel exited before fresh challenge response'); time.sleep(.025)
        response = q.read(proof)
        require(all(response.get(k) == v for k,v in challenge.items()), 'Foreign/replayed sentinel proof')
        match_fixture(response,admitted[1]['broker_ready'],commands[1],'sentinel')
        result['sentinel_survival'] = dict(challenge=challenge,response=response,host_monotonic_s=time.monotonic(),
                                         windows=runners[1].last_windows_observation)
        native_live(runners[1].last_windows_observation,time.monotonic())
        require(not futures[1].done(), 'Sentinel not alive before release')
        release = dict(challenge,action='RELEASE')
        q.save(commands[1].cwd/'release.json',release)
        sentinel = futures[1].result(timeout=33)
        result['sentinel'] = sentinel
        settled(sentinel,'succeeded',commands[1],admitted[1]['broker_ready'])
        success = q.read(commands[1].cwd/'success.json')
        require(all(success.get(k) == v for k,v in release.items()) and success.get('terminal') == 'succeeded',
                'Missing sentinel success receipt')
        match_fixture(success,admitted[1]['broker_ready'],commands[1],'sentinel')
        for i, mode in enumerate(('target','sentinel')):
            rows = records(out/(mode+'-process')/commands[i].stage_id/'protocol.jsonl',commands[i].context,commands[i].stage_id)
            require(len([x for x in rows if x['event']=='terminal']) == 1 and rows[-1]['event']=='linux-client-exited'
                    and rows[-1]['returncode']==0, 'Incomplete broker/client terminal protocol')
        require(attempt['source_pins'] == q.source_pins(), 'Source changed during pair')
        require(not any(x.split('.')[0] in ('torch','triton','flex_gemm') for x in sys.modules), 'Forbidden model import')
        result['classification'] = 'PASS'
    except BaseException as error:
        result['first_error'] = _error_detail(error,'coordinator.WSL-pair')
    finally:
        if result['classification'] != 'PASS':
            for event in events: event.set()
        for i,future in enumerate(futures):
            try:
                result.setdefault(('target','sentinel')[i],future.result(timeout=max(.001,attempt['grace_end']-time.monotonic())))
            except BaseException as error:
                result.setdefault('cleanup_errors',[]).append(_error_detail(error,'coordinator.cleanup'))
        pool.shutdown(wait=True)
        result['elapsed_s'] = time.monotonic()-began
        q.save(out/'pair-result.json',result)
    return 0 if result['classification']=='PASS' else 1
