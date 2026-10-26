"""Offline payload authentication regressions; never import installed runtimes."""
import argparse,base64,csv,hashlib,io,json,os,stat,sys,tempfile,unittest,warnings,zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c

TMP=None
CACHE=None
def encoded(data):return 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
def fixture(root,files=None,claims=None):
    files=files or {'tiny.py':b'qualified payload','tiny-1.dist-info/METADATA':b'Name: tiny\nVersion: 1\n'}
    record='tiny-1.dist-info/RECORD'
    rows=claims if claims is not None else [(n,encoded(b),str(len(b))) for n,b in files.items()]
    text=io.StringIO();csv.writer(text,lineterminator='\n').writerows([*rows,(record,'','')])
    archive=root/'tiny-1-py3-none-any.whl'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in files.items():z.writestr(name,data)
        z.writestr(record,text.getvalue())
    pin=dict(filename=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    env=root/'env';site=env/'lib/python3.12/site-packages';site.mkdir(parents=True)
    for n,b in {**files,record:text.getvalue().encode()}.items():
        p=site/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    return archive,pin,env,site

class Payload(unittest.TestCase):
    def test_stale_record_hash_and_size_are_reported_but_exact_payload_is_required(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d),claims=[('tiny.py',encoded(b'stale'),'5')])
            result=c.verify_installed_wheel(archive,env,site,expected=pin)
            self.assertEqual(result['files_verified'],2)
            self.assertTrue(any(r['path']=='tiny.py' and r['kind']=='metadata_differs' for r in result['RECORD_discrepancies']))
            (site/'tiny.py').write_bytes(b'qualified payloae')
            with self.assertRaises(RuntimeError):c.verify_installed_wheel(archive,env,site,expected=pin)

    def test_full_archive_tampering_is_rejected_before_ZIP_interpretation(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));raw=bytearray(archive.read_bytes());raw[0]^=1;archive.write_bytes(raw)
            with patch.object(zipfile,'ZipFile',side_effect=AssertionError('untrusted ZIP parsed')):
                with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,env,site)

    def test_nested_vendored_RECORD_is_verified_and_top_level_is_selected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d),files={'tiny/_vendor/other-2.dist-info/RECORD':b'vendored metadata','tiny.py':b'code'})
            result=c.verify_installed_wheel(archive,env,site,expected=pin)
            self.assertEqual(result['files_verified'],2)
            (site/'tiny/_vendor/other-2.dist-info/RECORD').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):c.verify_installed_wheel(archive,env,site,expected=pin)

    def test_record_only_native_names_never_substitute_for_literal_payload_names(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d),files={'tiny/libelf.so':b'ELF'},claims=[('tiny/libelf.so.1',encoded(b'ELF'),'3')])
            result=c.verify_installed_wheel(archive,env,site,expected=pin)
            self.assertEqual({(r['kind'],r['path']) for r in result['RECORD_discrepancies']},{('payload_absent_from_RECORD','tiny/libelf.so'),('RECORD_path_absent_from_payload','tiny/libelf.so.1')})
            (site/'tiny/libelf.so').rename(site/'tiny/libelf.so.1')
            with self.assertRaises(FileNotFoundError):c.verify_installed_wheel(archive,env,site,expected=pin)

    def test_missing_payload_and_missing_top_level_RECORD_are_rejected(self):
        for name in ('tiny.py','tiny-1.dist-info/RECORD'):
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=TMP) as d:
                archive,pin,env,site=fixture(Path(d));(site/name).unlink()
                with self.assertRaises(FileNotFoundError):c.verify_installed_wheel(archive,env,site,expected=pin)

    def test_pip_rewritten_top_level_RECORD_is_hashed_not_treated_as_runtime_bytes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));data=b'pip rewritten metadata';(site/'tiny-1.dist-info/RECORD').write_bytes(data)
            result=c.verify_installed_wheel(archive,env,site,expected=pin)
            self.assertEqual(result['installed_RECORD']['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(result['files_verified'],2)

    def test_unknown_installed_native_or_python_file_is_not_trusted_by_RECORD(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));trusted=set()
            c.verify_installed_wheel(archive,env,site,expected=pin,trusted_paths=trusted)
            for name in ('extra.py','extra.so'):
                path=site/name;path.write_bytes(b'unknown')
                with self.assertRaises(RuntimeError):c.verify_import_inventory(site,trusted,{'tiny-1.dist-info'})
                path.unlink()

    def test_archive_traversal_absolute_windows_alias_and_unsupported_spread_fail(self):
        for name in ('../outside','/absolute','a//b','a/./b','C:alias','a\\b','tiny-1.data/scripts/entry'):
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=TMP) as d:
                root=Path(d);archive=root/'tiny-1-py3-none-any.whl'
                with zipfile.ZipFile(archive,'w') as z:
                    item=zipfile.ZipInfo('placeholder');item.filename=name
                    z.writestr(item,b'unsafe');z.writestr('tiny-1.dist-info/RECORD','')
                pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                with self.assertRaises(ValueError):c.wheel_manifest(archive,pin,root/'env',root/'env/site')

    def test_ZIP_symlink_fifo_and_duplicate_members_fail(self):
        for mode in (stat.S_IFLNK|0o777,stat.S_IFIFO|0o600,None):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=TMP) as d:
                root=Path(d);archive=root/'tiny-1-py3-none-any.whl'
                with warnings.catch_warnings(),zipfile.ZipFile(archive,'w') as z:
                    warnings.simplefilter('ignore',UserWarning)
                    if mode:
                        item=zipfile.ZipInfo('tiny.py');item.create_system=3;item.external_attr=mode<<16;z.writestr(item,b'alias')
                    else:z.writestr('tiny.py',b'first');z.writestr('tiny.py',b'second')
                    z.writestr('tiny-1.dist-info/RECORD','')
                pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,root/'env',root/'env/site')

    def test_data_spread_and_cross_archive_destinations_cannot_overlap(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);archive,pin,env,site=fixture(root);claims={}
            c.wheel_manifest(archive,pin,env,site,claims=claims)
            with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,env,site,claims=claims)
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('tiny.py',b'one');z.writestr('tiny-1.data/purelib/tiny.py',b'two');z.writestr('tiny-1.dist-info/RECORD','')
            pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,env,site)

    def test_file_directory_prefix_collision_is_rejected_in_either_order(self):
        for names in (('tiny','tiny/code.py'),('tiny/code.py','tiny')):
            with self.subTest(names=names),tempfile.TemporaryDirectory(dir=TMP) as d:
                root=Path(d);archive=root/'tiny-1-py3-none-any.whl'
                with zipfile.ZipFile(archive,'w') as z:
                    for name in names:z.writestr(name,b'x')
                    z.writestr('tiny-1.dist-info/RECORD','')
                pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,root/'env',root/'env/site')

    def test_explicit_directories_reserve_cross_wheel_paths_and_reject_file_ancestors(self):
        for names in (('shared/',),('tiny.py','tiny.py/unused/')):
            with self.subTest(names=names),tempfile.TemporaryDirectory(dir=TMP) as d:
                root=Path(d);archive=root/'tiny-1-py3-none-any.whl';claims={}
                with zipfile.ZipFile(archive,'w') as z:
                    for name in names:z.writestr(name,b'')
                    z.writestr('tiny-1.dist-info/RECORD','')
                pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                if len(names)>1:
                    with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,root/'env',root/'env/site',claims=claims)
                else:
                    c.wheel_manifest(archive,pin,root/'env',root/'env/site',claims=claims)
                    archive=root/'other-2-py3-none-any.whl'
                    with zipfile.ZipFile(archive,'w') as z:z.writestr('shared',b'file');z.writestr('other-2.dist-info/RECORD','')
                    pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                    with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,root/'env',root/'env/site',claims=claims)

    def test_NUL_truncated_original_ZIP_name_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);archive=root/'tiny-1-py3-none-any.whl';env=root/'env';site=env/'site'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('fooQbar',b'x');z.writestr('tiny-1.dist-info/RECORD','')
            archive.write_bytes(archive.read_bytes().replace(b'fooQbar',b'foo\x00bar'))
            pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):c.wheel_manifest(archive,pin,env,site)

    def test_duplicate_RECORD_rows_and_multiple_top_level_distributions_fail(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);archive,pin,env,site=fixture(root,claims=[('tiny.py','',''),('tiny.py','','')])
            with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,env,site)
            with zipfile.ZipFile(archive,'a') as z:z.writestr('unrelated-2.dist-info/RECORD','')
            pin=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError):c.wheel_manifest(archive,pin,env,site)

    def test_real_installed_hardlink_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);archive,pin,env,site=fixture(root);os.link(site/'tiny.py',root/'alias')
            with self.assertRaises(ValueError):c.verify_installed_wheel(archive,env,site,expected=pin)

    def test_installed_symlink_and_parent_redirect_are_rejected(self):
        # Windows symlink creation can require privileges; inject only the OS-observed
        # type/identity and keep the production no-follow reader/path check real.
        import amd_pixal3d_wheel_payload as payload
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));path=site/'tiny.py';before=path.lstat()
            fake=os.stat_result((stat.S_IFLNK|0o777,*tuple(before)[1:]))
            original=Path.lstat
            with patch.object(Path,'lstat',lambda p,*a,**kw:fake if p==path else original(p,*a,**kw)):
                with self.assertRaises((RuntimeError,ValueError)):c.verify_installed_wheel(archive,env,site,expected=pin)
            original_link=Path.is_symlink
            with patch.object(Path,'is_symlink',lambda p:True if p==site else original_link(p)):
                with self.assertRaises(ValueError):c.verify_installed_wheel(archive,env,site,expected=pin)

    def test_read_time_installed_file_mutation_is_detected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            archive,pin,env,site=fixture(Path(d));manifest=c.wheel_manifest(archive,pin,env,site);changed=False
            def mutate():
                nonlocal changed
                if not changed:(site/'tiny.py').write_bytes(b'qualified payloae');changed=True
            with self.assertRaises(RuntimeError):c.verify_manifest(manifest,env,site,progress=mutate)

class PinnedWheels(unittest.TestCase):
    def test_all_13_authenticated_archives_and_literal_Triton_payloads(self):
        if not CACHE:self.skipTest('Supply --pinned-cache for the real 13-wheel offline audit')
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);cfg=c.lock();claims={};rows=[];triton=None;setuptools=None
            for pin in cfg['packages']:
                manifest=c.wheel_manifest(Path(CACHE)/pin['filename'],pin,root/'env',root/'env/site',claims=claims)
                rows.append(dict(archive=manifest['archive'],archive_identity=manifest['archive_identity'],files=len(manifest['files']),expanded_bytes=manifest['expanded_bytes'],RECORD_discrepancies=manifest['RECORD_discrepancies']))
                if pin['name']=='triton':triton=manifest
                if pin['name']=='setuptools':setuptools=manifest
            self.assertEqual(len(rows),13);self.assertEqual(sum(r['files'] for r in rows),25358)
            self.assertEqual(triton['files']['triton/_C/libproton.so'],dict(bytes=17923497,sha256='6dd1972835cf57ab43a0dfbbbb5046e2f1deed159ac553b2a0272df68716737f'))
            for name in ('libelf.so','libnuma.so','libtinfo.so'):self.assertIn('triton/backends/amd/lib/'+name,triton['files'])
            self.assertEqual(setuptools['record'],'setuptools-84.0.0.dist-info/RECORD')
            self.assertEqual(sum(n.endswith('.dist-info/RECORD') for n in setuptools['files']),13)
            if TMP:
                with (Path(TMP)/'pinned-wheel-audit.json').open('x',encoding='utf-8') as stream:json.dump(dict(archives=rows,collisions=0),stream,indent=2)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scratch',required=True);parser.add_argument('--pinned-cache');args=parser.parse_args()
    TMP=str(Path(args.scratch).resolve());CACHE=args.pinned_cache
    # The optional cached-wheel test is enabled only by an explicit local path.
    unittest.main(argv=[sys.argv[0]],verbosity=2)
