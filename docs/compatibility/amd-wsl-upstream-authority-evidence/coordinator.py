"""One host-owned synthetic conditioning ticket, never a complete neural job."""
import copy
import os
import sys
import threading
import time


def live_fixture(q,ready,runner,command,expected):
    for key,value in expected.items():
        if ready.get(key)!=value:raise ValueError('Foreign fixture binding: '+key)
    path=q.OUT/'runner/conditioning/protocol.jsonl'
    if not path.exists():return None
    data=path.read_bytes()
    if len(data)>1024**2:raise ValueError('Bounded diagnostic gate exceeded')
    rows=[q.json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b'\n')]
    rows=[row for row in rows if row.get('event')=='leader-executed']
    if not rows:return None
    if len(rows)!=1:raise ValueError('Repeated execution observation')
    execution=rows[0]
    if any(execution.get(k)!=expected[k] for k in ('context','stage_id','nonce')):raise ValueError('Foreign diagnostic gate')
    if (ready.get('uid')!=1000 or ready.get('euid')!=1000 or ready.get('user')!='modly'
            or ready.get('executable')!='/usr/bin/python3.12' or ready.get('rlimit_as')!=[6*1024**3]*2
            or any(ready.get('identity',{}).get(k)!=execution['leader'].get(k) for k in ('pid','ppid','uid','start_ticks','session','pgrp'))):
        raise ValueError('Actual Linux fixture differs from native exec gate')
    return dict(fixture=ready,execution_diagnostic=execution,windows=runner.last_windows_observation,
                scope='Release timing only; private runner pipe and local validator issue authority independently')


def run(q):
    from services.amd_completed_result import CompletionContext, FilePin
    from services.amd_upstream_job import UpstreamJob, StageSpec, OutputSpec
    from services.amd_stage_process import OwnedStageRunner, StageCommand, _error_detail, _guest_path
    from services.amd_windows_authority import consume
    OUT=q.OUT
    q.save(OUT/'worker-ready.json',dict(pid=os.getpid(),monotonic_s=time.monotonic()))
    if sys.stdin.buffer.readline(32)!=b'GO\n':raise ValueError('Outer Job GO required')
    attempt=q.read(OUT/'attempt.json')
    if attempt['source_pins']!=q.source_pins() or attempt['interpreter_sha256']!=q.sha(sys.executable):
        raise ValueError('Reviewed source changed')
    result=dict(classification='BLOCKED',synthetic_conditioning_only=True,genuine_neural_qualified=False,
                complete_shape_job=False,worker_invocations=1,negative_checks={})
    cancel=threading.Event(); began=time.monotonic()
    try:
        private=OUT/'host-tickets';private.mkdir()
        root=OUT/'synthetic-evidence';root.mkdir()
        leaf=root/'conditioning';leaf.mkdir()
        shape=root/'shape';shape.mkdir()
        params={'purpose':'explicit-synthetic-WSL-authority-only'}
        image=b'EXPLICIT SYNTHETIC INPUT - NO IMAGE OR NEURAL COMPUTATION'
        context=CompletionContext.new('synthetic-authority',image,params,mode='generated',model_id='trellis2-amd')
        imagepath=root/'synthetic-image.bin';imagepath.write_bytes(image)
        contract=root/'input-contract.json'
        q.save(contract,dict(source_sha256=context.input_sha256,synthetic=True))
        output=leaf/'synthetic-output.json'
        stages=(StageSpec('conditioning',(OutputSpec(output,65536),)),
                StageSpec('occupancy',(OutputSpec(root/'occupancy'/'unused.json',65536),)),
                StageSpec('shape',tuple(OutputSpec(shape/'run'/n,65536) for n in
                    ('results.json','scratch/shape-latent-denormalized.npy','scratch/shape-latent-coordinates.npy'))))
        job=UpstreamJob(private,context=context,image_bytes=image,params=params,shape_evidence=shape,
            source_pins={'source'+str(i):FilePin(q.ROOT/name,digest) for i,(name,digest) in enumerate(q.source_pins().items())},
            runtime_input_pins={'image':FilePin(imagepath,q.sha(imagepath)),'input_contract':FilePin(contract,q.sha(contract))},
            stages=stages,cancel_event=cancel)
        ticket=job.issue('conditioning');runner=OwnedStageRunner(OUT/'runner')
        command=StageCommand('conditioning','wsl',('/usr/bin/python3','-B','-I','-S',_guest_path(q.FIXTURE),_guest_path(leaf)),
                             leaf,context.to_dict(),25)
        binding=job.bind_wsl(ticket,runner,command)
        released=False
        def check():
            nonlocal released
            if (OUT/'host-cancel.json').exists() or time.monotonic()>=attempt['active_end']:
                cancel.set();raise InterruptedError('Outer cancellation/deadline')
            ready=leaf/'ready.json'
            if ready.exists() and not (leaf/'ready.json.writing').exists() and not released:
                expected=dict(context=context.to_dict(),stage_id='conditioning',nonce=ticket.nonce,synthetic=True)
                proof=live_fixture(q, q.read(ready),runner,command,expected)
                if proof is not None:
                    result['live_fixture_identity']=proof
                    q.save(leaf/'release.json',expected);released=True
        receipt,authority=runner.run_bound(binding,tick=check)
        result['runner']=receipt
        if authority is None:raise ValueError('Actual runner issued no completion authority')
        for name,bad_ticket,bad_authority in (
                ('copied-authority',ticket,copy.copy(authority)),
                ('copied-ticket',copy.copy(ticket),authority),
                ('worker-PASS',ticket,dict(terminal='succeeded',natural=True,settled=True))):
            try:consume(bad_authority,job,bad_ticket,runner,backendkind='wsl')
            except (ValueError,TypeError):result['negative_checks'][name]=True
            else:raise ValueError('Copied/foreign authority accepted: '+name)
        accepted=job.accept_wsl(ticket,runner,authority)
        result['accepted_receipt']=dict(path=str(accepted.path),sha256=accepted.sha256,document=q.read(accepted.path))
        try:consume(authority,job,ticket,runner,backendkind='wsl')
        except (ValueError,TypeError):result['negative_checks']['replay']=True
        else:raise ValueError('Consumed proof replay accepted')
        try:runner.run_bound(binding)
        except ValueError:result['negative_checks']['repeated-invocation']=True
        else:raise ValueError('Invocation replayed')
        expected=dict(context=context.to_dict(),stage_id='conditioning',nonce=ticket.nonce,synthetic=True,
                      value='synthetic conditioning transport only; no model evidence')
        if q.read(output)!=expected:raise ValueError('Synthetic output differs')
        if q.read(accepted.path).get('qualification')!='WSL_CPU_SUPERVISED' or q.read(accepted.path).get('genuine_execution_qualified') is not False:raise ValueError('Wrong scope')
        if len(job._accepted)!=1 or job._completion is not None:raise ValueError('Unexpected complete neural claim')
        if any(n.split('.')[0] in ('torch','triton','flex_gemm') for n in sys.modules):raise ValueError('Forbidden runtime')
        check()
        result.update(classification='PASS',output_sha256=q.sha(output),accepted_stages=1,
                      product_context=context.to_dict(),ticket_nonce=ticket.nonce)
    except BaseException as error:
        cancel.set();result['error']=_error_detail(error,'coordinator.bridge')
    result['elapsed_s']=time.monotonic()-began
    q.save(OUT/'pair-result.json',result)
    return 0 if result['classification']=='PASS' else 1
