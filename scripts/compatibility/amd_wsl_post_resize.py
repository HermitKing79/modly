"""One held read-only post-resize capacity check. Default is inert."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import amd_wsl_resize_operator as o
import amd_wsl_post_resize_geometry as g
b=o.b;r=o.r
CASE=b.ROOT.parent/'artifacts/amd-wsl-post-resize-capacity-v2'
FAILED_CASE=b.ROOT.parent/'artifacts/amd-wsl-post-resize-capacity-v1'
PRIOR=b.ROOT.parent/'artifacts/amd-wsl-resize-64g-v1'
OLD_AUTH=b.ROOT.parent/'artifacts/amd-wsl-post-resize-preparation-20261030-v1/authority.json'
AUTH=b.ROOT.parent/'artifacts/amd-wsl-post-resize-v2-preparation-20261030-v1/authority.json'
AUTH_SHA='7a4c00ba7b6cf4a2fa862121875d003d1ce62841f76d15395d2d9a355179b068'
GUEST=Path(__file__).with_name('amd_wsl_post_resize_capacity.py')
FILES=['scripts/compatibility/'+x for x in ('amd_wsl_post_resize.py','amd_wsl_post_resize_geometry.py','amd_wsl_post_resize_capacity.py','Invoke-ModlyPostResizeCapacity.ps1')]

def post_v1_source(previous,fresh):
    r.require(fresh['metadata']==previous['metadata'],'Post-v1 container metadata changed')
    identity=lambda row:{k:v for k,v in row['identity'].items() if k!='mtime_ns'}
    r.require(identity(fresh)==identity(previous),'Post-v1 non-timestamp source identity changed')
    r.require(type(fresh['identity']['mtime_ns']) is int and fresh['identity']['mtime_ns']>=previous['identity']['mtime_ns'],'Post-v1 timestamp invalid')
    return fresh

def authenticate(a):
    r.require(a['predecessor_commit']=='28ba485e2756ef00de8fc1d7515bd6ef12252703' and
              b.digest(OLD_AUTH)==a['previous_authority_sha256']=='aeaadd28a745d449b4b6541f8cd8ba350207dd696543b73f3939e8bab1224ed1','Previous authority differs')
    r.require({p.name for p in FAILED_CASE.iterdir()}==set(a['v1_failure_receipts'])|set(a['v1_empty_directories']),'Complete v1 evidence set differs')
    for name,expected in a['v1_empty_directories'].items():
        path=b.plain(FAILED_CASE/name);s=path.stat()
        r.require(path.is_dir() and not any(path.iterdir()) and dict(file_id=s.st_ino,mtime_ns=s.st_mtime_ns)==expected,'Preserved v1 empty directory differs')
    for parent,key in ((PRIOR,'resize_receipts'),(o.ACCEPTED,'backup_receipts'),(FAILED_CASE,'v1_failure_receipts')):
        for name,sha in a[key].items():r.require(b.digest(parent/name)==sha,'Historical evidence differs: '+name)
    for name,sha in a['native_binaries'].items():r.require(b.digest(Path(name))==sha,'Native binary differs')
    load=lambda name:json.loads((PRIOR/name).read_text())
    final=load('final-result.json');settled=load('settled-container-state.json')
    events=[json.loads(x) for x in (PRIOR/'events.jsonl').read_text().splitlines()]
    native=[x for x in events if x['event']=='NATIVE_SETTLED']
    r.require(final['commit']=='d0779b83f0c3ef50ea7aeb4ea64aa8f8a81af811' and final['native_settled'] is True and len(native)==1 and native[0]['returncode']==0,'Natural resize completion unverified')
    r.require(settled['metadata']['virtual_capacity_bytes']==r.TARGET_BYTES,'Saved container capacity differs')
    return post_v1_source(settled,a['source_after_v1']),load('cold-identities.json')

def publication(commit):
    o.publication(commit)
    for name in FILES:
        saved=subprocess.check_output(['git','-C',str(b.ROOT),'show',commit+':'+name],timeout=15)
        r.require(saved.replace(b'\r\n',b'\n')==(b.ROOT/name).read_bytes().replace(b'\r\n',b'\n'),'Published capacity source differs')

def query(ob):
    ob.check();previous=(o.CASE,o.GUEST)
    try:
        o.CASE=CASE;o.GUEST=GUEST
        return o.capacity_query(ob)
    finally:o.CASE,o.GUEST=previous

def final_inventory(ob,a):
    old=b.CASE;b.CASE=CASE
    try:
        c=b.Controller();c.health=lambda:ob.check()
        row=json.loads(c.child(b.helper_command('inventory'),'after-inventory',30))
    finally:b.CASE=old
    b.save(CASE/'after-inventory.json',row)
    r.require(set(row['running'])<={'Modly-AMD-Ubuntu-24.04'},'Unrelated WSL activity appeared')
    o.docker_quiet();b.authenticate_inventory(dict(row,running=[],vm_processes=[]),a)

def run(commit):
    publication(commit);b.authenticate_worker_python();o.no_inherited_job()
    r.require(b.digest(AUTH)==AUTH_SHA,'Capacity authority differs');auth=json.loads(AUTH.read_text());a=b.authority()
    settled,cold=authenticate(auth)
    b.plain(CASE,False);r.require(not CASE.exists(),'Consumed evidence path; no retries')
    pins=b.Pins();created=False;old=(o.CASE,o.WINDOW);o.CASE=CASE;o.WINDOW=120;ob=o.Observer()
    report=dict(classification='FAIL',measurement='UNAVAILABLE',started=o.utc(),commit=commit,no_resize=True,no_decoder=True)
    try:
        pins.add(CASE.parent);CASE.mkdir();created=True;pins.add(CASE)
        ob.preflight();o.journal('BEGIN_READ_ONLY_CAPACITY')
        o.inventory(ob,'initial-inventory',a);b.save(CASE/'inputs-before.json',b.inputs(a))
        pins.add(o.BACKUP.parent);pins.add(b.SOURCE.parent)
        with b.native_file(o.BACKUP,'verify') as backup:
            backup_id=b.file_identity(backup);backup_meta=b.metadata(backup)
            r.require(dict(identity=backup_id,metadata=backup_meta)==cold['backup'],'Accepted backup identity/metadata differs')
            load=lambda name:json.loads((o.ACCEPTED/name).read_text())
            r.verify_backup(load('final-result.json'),load('worker-result.json'),load('destination-hash-stdout.txt'),
                [load(x) for x in ('inventory-process.json','backup-worker-process.json','final-inventory-process.json')],cold)
            with b.native_file(b.SOURCE,'verify') as source:
                now=dict(identity=b.file_identity(source),metadata=b.metadata(source))
                r.require(now==settled,'Existing resized source differs; review before startup')
                b.save(CASE/'source-before.json',now)
            for i in range(3):
                sample=ob.check();o.journal('ADMISSION',index=i,**sample)
                if i<2:time.sleep(4)
            o.journal('FINAL_BEFORE_QUERY',**ob.check())
            row=query(ob);assessment=g.assess(row)
            final_inventory(ob,a)
            r.require(b.file_identity(backup)==backup_id and b.metadata(backup)==backup_meta,'Backup changed')
            s=b.plain(b.SOURCE).stat();r.require(s.st_ino==settled['identity']['file_id'] and s.st_nlink==1,'Backing file identity changed')
            b.save(CASE/'source-after-stat.json',dict(file_id=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns))
            b.save(CASE/'inputs-after.json',b.inputs(a));authenticate(auth);ob.check()
            report.update(measurement='PASS',assessment=assessment,classification='PASS_SPACE_GATE' if assessment['recommendation']=='GO_FOR_DECODE_PREPARATION_ONLY' else 'NO_GO_CAPACITY')
    except BaseException as exc:
        ob.faults.add(type(exc).__name__+': '+str(exc));report.update(classification='FAIL',error=type(exc).__name__+': '+str(exc))
    finally:
        try:pins.close();report['handles_closed']=True
        except BaseException as exc:ob.faults.add('Cleanup: '+str(exc))
        if ob.faults.rows:report['classification']='FAIL'
        report.update(faults=ob.faults.rows,seconds=time.monotonic()-ob.began,minimum_available_bytes=ob.minimum,no_retry=True)
        if created:
            try:
                with (CASE/'REPORT.txt').open('x',encoding='utf-8') as f:
                    f.write('Only final-result.json is authoritative.\n'+json.dumps(report,indent=2)+'\n');f.flush();os.fsync(f.fileno())
                b.save(CASE/'final-result.pending.json',report)
                os.rename(CASE/'final-result.pending.json',CASE/'final-result.json')
            except BaseException as exc:ob.faults.add('Publication: '+str(exc));report['classification']='FAIL'
        o.CASE,o.WINDOW=old
        print('CAPACITY='+report['classification']+'\nEVIDENCE='+str(CASE),flush=True)
    return 0 if report['classification']=='PASS_SPACE_GATE' else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--approve-capacity-check',action='store_true');p.add_argument('--commit');args=p.parse_args(argv)
    if not args.run:print('DRY_RUN: no admission, disk access, WSL or resize. Execution HOLD.');return 0
    if not args.approve_capacity_check or not re.fullmatch('[a-f0-9]{40}',args.commit or ''):p.error('Separate capacity-check approval and exact commit required')
    return run(args.commit)

if __name__=='__main__':raise SystemExit(main())
