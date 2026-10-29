"""Proposed operator-only admission; original neural contract remains unchanged."""
import hashlib,json,os,re,sys,time,zipfile
from pathlib import Path
import amd_pixal3d_proof_common_v2 as v2
import amd_pixal3d_kitchen_overlay as overlay
from amd_pixal3d_proof_common_v2 import ROOT,GIB,OPERATORS,now,digest,plain,inside,save,journal,require_resources,verify,CacheReadLease
CONTRACT=ROOT/'docs/compatibility/amd-pixal3d-operator-v3-contract.json'

def lock():
    proposed=json.loads(CONTRACT.read_text());base=v2.lock()
    if proposed['schema']!='modly.pixal3d.operator-only.proposed.v3':raise ValueError('Unknown operator profile')
    if hashlib.sha256(v2.CONTRACT.read_text().encode()).hexdigest()!=proposed['v2_contract_canonical_sha256']:raise ValueError('V2 contract changed')
    if proposed['execution_authorized_in_chat'] or proposed['generation_authorized']:raise ValueError('Preparation-only proposal required')
    if proposed.get('runtime_overlay')!=overlay.PIN:raise ValueError('Exact authenticated packaging overlay required')
    if proposed.get('operator_execution_hold') is not True:raise ValueError('Unproven audit trigger requires execution hold')
    value=dict(base,**proposed);p=value['operator_policy']
    ready=base['guards']['live_host_RAM_floor_bytes']+base['guards']['CPU_postprocess']['commit_bytes']
    if p['ready_bytes']!=ready or p['observed_startup_decline_bytes']!=2360119296 or p['prelaunch_bytes']!=ready+2360119296:raise ValueError('Measured budget derivation changed')
    if value['guest_root']!='/home/modly/pixal3d-native-proof-v2' or value['temporary_guest_root']!='/home/modly/pixal3d-operator-proof-v3-kitchen-overlay' or value['windows_root']!='D:/AI/Modly-Pixal3D-Operators-v3-kitchen-overlay':raise ValueError('Exact operator-only roots required')
    return value

def require_profile(row,floor):
    if type(row.get('available_bytes')) is not int or row['available_bytes']<floor:raise RuntimeError('Proposed model-free RAM admission failed: '+str(floor)+' bytes required')

def admit_operator(read,sleep,record,execute):
    floor=lock()['operator_policy']['prelaunch_bytes']
    for index in range(3):
        row=dict(read(),timestamp_utc=now());record(row);require_profile(row,floor)
        if index<2:sleep(4)
    row=dict(read(),timestamp_utc=now(),stage='final-before-operator-controller');record(row);require_profile(row,floor)
    return execute()

def evidence_document(cfg,name):
    row=cfg['evidence_files'][name];path=inside(ROOT.parent/'artifacts'/cfg['v2_evidence_case'],name)
    verify(path,row);return json.loads(path.read_text())

def validate_v2_evidence(cfg):
    summary=evidence_document(cfg,'summary.json');host=evidence_document(cfg,'provision/host-result.json');child=evidence_document(cfg,'provision/child-result.json')
    if summary.get('commit')!='681fcc962d4ca904291cc3b7bfbdf28757350ba6' or summary.get('stages',{}).get('provisioning',{}).get('status')!='PASS':raise RuntimeError('Successful v2 provisioning required')
    if host.get('classification')!='PASS' or not host.get('GO_sent') or host.get('exit_code')!=0 or not host.get('guest_cleanup',{}).get('natural_exit') or host.get('Windows_job_cleanup',{}).get('forced_job_cleanup') is not False:raise RuntimeError('V2 provisioning/cleanup receipt failed')
    expected={p['name'].lower().replace('_','-'):p['version'] for p in cfg['packages']}
    actual={k.lower().replace('_','-'):v for k,v in child.get('runtime_packages',{}).items()}
    if child.get('classification')!='PASS' or actual!=expected or child.get('share_site_packages') is not False or child.get('isolated_prefix')!=cfg['guest_root']+'/.venv':raise RuntimeError('V2 dependency closure differs')
    if child.get('cache_before')!=child.get('cache_after') or child.get('baseline_critical_files_before')!=child.get('baseline_critical_files_after'):raise RuntimeError('V2 preservation failed')
    if child['baseline_critical_files_after']['lib/python3.12/site-packages/torch/version.py']!=cfg['runtime']['torch_version_file_sha256']:raise RuntimeError('Qualified runtime identity differs')
    return dict(status='PASS',packages=len(actual),pip_install_completed=True,natural_cleanup=True,operator_execution='NOT_RUN')

from amd_pixal3d_wheel_payload import installed_path,verify_installed_wheel,wheel_manifest,verify_manifest,AuthenticationProgress


def verify_import_inventory(site,trusted_paths,dist_infos):
    bootstrap=0;generated={'INSTALLER','REQUESTED','direct_url.json','RECORD'}
    for path in site.rglob('*'):
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise RuntimeError('Unapproved installed link/alias: '+str(path.relative_to(site)))
        if not path.is_file():continue
        name=path.relative_to(site).as_posix();parts=name.split('/')
        if name in trusted_paths:continue
        if len(parts)==2 and parts[0] in dist_infos and parts[1] in generated:continue
        if parts[0]=='pip' or re.fullmatch(r'pip-[0-9][A-Za-z0-9.!+_-]*\.dist-info',parts[0]):
            bootstrap+=1;continue # existing ensurepip bootstrap; operator audit prohibits importing pip.
        raise RuntimeError('Unapproved installed/importable file: '+name)
    return dict(unknown_files=0,bootstrap_files_excluded_from_operator_imports=bootstrap)

def validate_venv_config(text,version):
    values={}
    for line in text.splitlines():
        if not line.strip():continue
        key,separator,value=line.partition('=');key=key.strip().lower();value=value.strip()
        if not separator or key in values:raise RuntimeError('Ambiguous venv configuration')
        values[key]=value
    expected={'home':'/usr/bin','include-system-site-packages':'false','version':version,'executable':'/usr/bin/python'+'.'.join(version.split('.')[:2])}
    if any(values.get(k)!=v for k,v in expected.items()) or set(values)-set(expected)-{'command'}:raise RuntimeError('Qualified isolated venv configuration differs')
    return dict(system_site_packages=False,home=values['home'],version=version,executable=values['executable'])

def environment_snapshot(root):
    root=plain(root);rows={}
    for path in root.rglob('*'):
        s=path.lstat();name=path.relative_to(root).as_posix()
        if path.is_symlink():
            path.resolve().relative_to(root);rows[name]=dict(link=os.readlink(path),mtime_ns=s.st_mtime_ns,file_id=s.st_ino)
        elif path.is_file():rows[name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,file_id=s.st_ino,mode=s.st_mode)
    return rows

def authenticate_environment(cfg,progress=None,archive_report=None):
    progress=progress if isinstance(progress,AuthenticationProgress) else AuthenticationProgress(progress)
    progress.enter('preflight',None,None,'runtime-identity')
    if os.name!='posix' or sys.version.split()[0]!=cfg['runtime']['python'] or sys.prefix!='/usr':raise RuntimeError('Qualified Linux system Python required')
    progress.checkpoint='validate-v2-receipts-and-roots'
    validate_v2_evidence(cfg);root=plain(cfg['guest_root']);env=plain(root/'.venv');site=env/'lib/python3.12/site-packages'
    progress.enter('preflight',None,'pyvenv.cfg','validate-venv-config')
    configuration=validate_venv_config((env/'pyvenv.cfg').read_text(),cfg['runtime']['python'])
    progress.enter('preflight',None,'bin/python','compare-copied-interpreter')
    if plain(env/'bin/python').read_bytes()!=Path('/usr/bin/python3').read_bytes():raise RuntimeError('Qualified copied interpreter differs')
    progress.enter('preflight',None,'pip-report.json','validate-pip-report')
    report=json.loads((root/'pip-report.json').read_text());expected={p['name'].lower().replace('_','-'):p for p in cfg['packages']}
    actual={p['metadata']['name'].lower().replace('_','-'):p for p in report['install']}
    if set(actual)!=set(expected):raise RuntimeError('Installed pip-report closure changed')
    installed=[];manifests=[];claims={};trusted_paths=set();cache=Path('/mnt/d/AI/Modly-Pixal3D-Proof-v1/downloads')
    for name,p in expected.items():
        progress.enter('archive-manifests','archive',p['filename'],'validate-pip-package-record',package=name)
        record=actual[name]
        if record['metadata']['version']!=p['version'] or record['download_info']['archive_info']['hashes']['sha256']!=p['sha256']:raise RuntimeError('Installed report identity differs')
        progress.checkpoint='locate-pinned-archive'
        archive=inside(cache,p['filename'])
        manifests.append((name,wheel_manifest(archive,p,env,site,progress,claims)))
        manifest=manifests[-1][1]
        progress.checkpoint='report-archive-manifest'
        if archive_report:archive_report(dict(package=name,archive=manifest['archive'],archive_identity=manifest['archive_identity'],
            files=len(manifest['files']),expanded_bytes=manifest['expanded_bytes'],top_level_RECORD=manifest['record'],
            RECORD_identity=manifest['files'][manifest['record']],RECORD_discrepancies=manifest['RECORD_discrepancies']))
    progress.package=None;progress.enter('archive-manifests',None,None,'aggregate-expanded-payload-budget')
    if sum(m['expanded_bytes'] for _,m in manifests)>20*GIB:raise RuntimeError('Original isolated growth budget exceeded')
    for name,manifest in manifests:
        progress.enter('installed-files','installed',None,'installed-package-start',package=name)
        installed.append(dict(package=name,**verify_manifest(manifest,env,site,progress,trusted_paths)))
    progress.package=None;progress.enter('inventory',None,None,'installed-import-inventory')
    dist_infos={n.split('/')[0] for n in trusted_paths if n.split('/')[0].endswith('.dist-info')}
    inventory=verify_import_inventory(site,trusted_paths,dist_infos)
    for row in cfg['sources']:
        progress.enter('source-integrity',None,row['path'],'source-sha256');verify(inside(root/'ComfyUI',row['path']),row)
    # No arbitrary .pth/startup module/bytecode may execute from outside trusted wheel inventories.
    progress.enter('startup-checks',None,'*.pth','startup-path-inventory')
    allowed_pth=set()
    for p in cfg['packages']:
        progress.enter('startup-checks',None,p['filename'],'enumerate-approved-startup-paths',package=p['name'])
        with zipfile.ZipFile(cache/p['filename']) as z:allowed_pth.update(n for n in z.namelist() if n.endswith('.pth'))
    progress.package=None;progress.enter('startup-checks',None,'*.pth','compare-startup-path-set')
    if {p.relative_to(site).as_posix() for p in site.glob('*.pth')}!=allowed_pth:raise RuntimeError('Unapproved Python startup path hook')
    for name in ('sitecustomize.py','usercustomize.py'):
        progress.enter('startup-checks',None,name,'forbidden-startup-module')
        if (site/name).exists():raise RuntimeError('Unapproved Python startup module')
    progress.enter('startup-checks',None,'*.pyc','runtime-bytecode-inventory')
    if any(p for p in site.rglob('*.pyc') if 'pip' not in p.relative_to(site).parts):raise RuntimeError('Unauthenticated runtime bytecode; no removal or repair')
    progress.enter('overlay','archive',overlay.PIN['filename'],'overlay-authentication',package='packaging')
    dependency_overlay=overlay.authenticate(progress);progress.checkpoint='authentication-complete';progress.finish('complete')
    return dict(status='PASS',packages=installed,files_verified=sum(r['files_verified'] for r in installed),bytes_verified=sum(r['bytes_verified'] for r in installed),
                env=str(env),writes=0,provisioning=False,downloads=False,import_inventory=inventory,configuration=configuration,dependency_overlay=dependency_overlay,authentication_progress=progress.snapshot())
