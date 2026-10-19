"""Recover exact CPU nomination source snapshots, checked against owned markers."""
import hashlib,json
from pathlib import Path
E=Path(__file__).resolve().parent;P=E.parents[2].parent/'artifacts/amd-brick-uv-bake-18c0c92'
D=P/'selection-failed-candidate';D.mkdir(exist_ok=True)

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    marker=json.loads((P/'bake-attempt.json').read_text())
    rows={Path(k).name:v for k,v in marker['source_before'].items() if Path(k).parent==E and Path(k).suffix=='.py'}
    for name,row in rows.items():
        target=D/name
        if target.exists():
            assert sha(target.read_bytes())==row['sha256'];continue
        data=(E/name).read_text()
        data=data.replace("artifacts/amd-brick-uv-bake-18c0c92/qualified","artifacts/amd-brick-uv-bake-18c0c92")
        if name=='patch_math.py':
            start=data.index('    if np.all(areas < -1e-14):')
            end=data.index('    return uv',start)
            data=data[:start]+"    if np.any(areas <= 1e-14):\n        raise ValueError('Front-facing positive-area injective chart required')\n"+data[end:]
        if name=='test_patch_math.py':
            start=data.index('        reversed_faces=self.f[:,::-1].copy()')
            end=data.index('        with self.assertRaises(ValueError):chart(self.v,np.array',start)
            data=data[:start]+"        with self.assertRaises(ValueError):chart(self.v,self.f[:,::-1].copy())\n"+data[end:]
        candidates=[data.encode(),data.replace('\n','\r\n').encode()]
        raw=next((b for b in candidates if sha(b)==row['sha256']),None)
        assert raw is not None,'Exact marker hash recovery failed: '+name
        with target.open('xb') as f:f.write(raw)
    initial=(D/'run_host.py').read_text()
    observed=initial.replace("choices=['bake','export','compare']","choices=['bake','nomination','export','compare']").replace("if args.stage!='bake':","if args.stage not in ('bake','nomination'):").replace("if args.stage=='bake':","if args.stage in ('bake','nomination'):").replace("str(E/'bake_patch.py')","str(E/('bake_patch.py' if args.stage=='bake' else 'inspect_nomination.py'))")
    expected=json.loads((P/'nomination-attempt.json').read_text())['source_before'][str(E/'run_host.py')]['sha256']
    raw=next((b for b in (observed.encode(),observed.replace('\n','\r\n').encode()) if sha(b)==expected),None)
    assert raw is not None,'Nomination host exact hash'
    with (D/'nomination_host.py').open('xb') as f:f.write(raw)
    print('All '+str(len(rows))+' initial sources and actual nomination host recovered byte-exact to pre-execution marker hashes')

if __name__=='__main__':main()
