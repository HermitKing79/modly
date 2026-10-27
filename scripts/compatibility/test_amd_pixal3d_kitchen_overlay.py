"""CPU-only real packaging tests; no mocked Torch or Kitchen implementations."""
import ast,email,hashlib,importlib,json,os,subprocess,sys,tempfile,unittest,zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import amd_pixal3d_operator_common_v3 as c
import amd_pixal3d_kitchen_overlay as overlay

TMP=None

@contextmanager
def active():
    saved={k:v for k,v in sys.modules.items() if k=='packaging' or k.startswith('packaging.')};old=list(sys.path)
    try:
        for k in saved:del sys.modules[k]
        yield overlay.activate()
    finally:
        sys.path[:]=old
        for k in list(sys.modules):
            if k=='packaging' or k.startswith('packaging.'):del sys.modules[k]
        sys.modules.update(saved)

class Overlay(unittest.TestCase):
    def test_missing_import_is_reproduced_then_real_pinned_ZIP_satisfies_it(self):
        saved={k:v for k,v in sys.modules.items() if k=='packaging' or k.startswith('packaging.')};old=list(sys.path)
        try:
            for k in saved:del sys.modules[k]
            sys.path[:]=[p for p in old if p!=str(overlay.archive_path())]
            with self.assertRaises(ModuleNotFoundError):importlib.import_module('packaging')
            report=overlay.activate()
            from packaging import version
            self.assertEqual(version.parse('2.9.1+rocm7.2.1.gitff65f5bc'),version.Version('2.9.1+rocm7.2.1.gitff65f5bc'))
            self.assertLess(version.parse('2.9.1'),version.parse('2.10.0'))
            self.assertEqual(report['version'],'25.0');self.assertEqual(report['installations'],0)
            self.assertEqual(report['archive_sha256'],'29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484')
        finally:
            sys.path[:]=old
            for k in list(sys.modules):
                if k=='packaging' or k.startswith('packaging.'):del sys.modules[k]
            sys.modules.update(saved)

    def test_every_real_packaging_module_imports_and_parses_without_other_packages(self):
        with active():
            with zipfile.ZipFile(overlay.archive_path()) as z:
                names=[n[:-3].replace('/','.') for n in z.namelist() if n.startswith('packaging/') and n.endswith('.py') and not n.endswith('/__init__.py')]
            for name in names:importlib.import_module(name)
            from packaging.requirements import Requirement
            from packaging.specifiers import SpecifierSet
            self.assertEqual(Requirement('torch>=2.9.0').name,'torch')
            self.assertTrue(SpecifierSet('>=2.9,<2.10').contains('2.9.1'))
            self.assertGreaterEqual(len(overlay.verify_loaded()),14)
            self.assertNotIn('torch',sys.modules)

    def test_preloaded_real_packaging_cannot_be_silently_replaced(self):
        with active():
            before=list(sys.path)
            with self.assertRaises(RuntimeError):overlay.activate()
            self.assertEqual(sys.path,before)

    def test_foreign_module_origin_is_rejected_without_replacing_real_module(self):
        with active():
            from packaging import version
            original=version.__file__
            try:
                version.__file__='/foreign/packaging/version.py'
                with self.assertRaises(RuntimeError):overlay.verify_loaded()
            finally:version.__file__=original

    def test_missing_or_tampered_wheel_stops_before_ZIP_parsing_or_path_activation(self):
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p=Path(d)/overlay.PIN['filename'];data=bytearray(overlay.archive_path().read_bytes());data[0]^=1;p.write_bytes(data);before=list(sys.path)
            with patch.object(overlay,'archive_path',return_value=p),patch.object(zipfile,'ZipFile',side_effect=AssertionError('untrusted ZIP parsed')):
                with self.assertRaises(RuntimeError):overlay.authenticate()
                p.unlink()
                with self.assertRaises(FileNotFoundError):overlay.authenticate()
            self.assertEqual(sys.path,before)

    def test_overlay_version_comparison_matches_exact_Kitchen_expression(self):
        pin=next(p for p in c.lock()['packages'] if p['name']=='comfy-kitchen');p=Path(c.lock()['cache_root'])/'downloads'/pin['filename'];c.verify(p,pin)
        with zipfile.ZipFile(p) as z:tree=ast.parse(z.read('comfy_kitchen/scaled_mm_v2.py'))
        expressions={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        # Use the literal qualified version string in place of the expression's
        # Torch attribute; this evaluates real packaging, without a fake Torch.
        class LiteralTorchVersion(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='__version__':return ast.copy_location(ast.Constant('2.9.1+rocm7.2.1.gitff65f5bc'),node)
                return self.generic_visit(node)
        with active():
            from packaging import version
            expression=ast.fix_missing_locations(ast.Expression(LiteralTorchVersion().visit(expressions['_TORCH_VERSION'])))
            actual=eval(compile(expression,'authenticated Kitchen version guard','eval'),{'version':version})
            self.assertEqual(str(actual),'2.9.1');self.assertLess(actual,version.parse('2.10.0'))

    def test_full_Kitchen_and_packaging_static_import_audit_has_only_one_missing_requirement(self):
        cfg=c.lock();pin=next(p for p in cfg['packages'] if p['name']=='comfy-kitchen');p=Path(cfg['cache_root'])/'downloads'/pin['filename']
        result=overlay.audit_imports(p,pin)
        self.assertEqual(result['required_third_party'],['packaging','torch'])
        self.assertGreaterEqual(result['python_files'],64)
        optional={row['module'] for row in result['guarded_optional_backend_imports']}
        self.assertEqual(optional,{'triton','torch_npu','nvidia','_manylinux'})
        with zipfile.ZipFile(p) as z:meta=email.message_from_bytes(z.read('comfy_kitchen-0.2.40.dist-info/METADATA'))
        self.assertTrue(all('extra ==' in row for row in meta.get_all('Requires-Dist',[])))
        if TMP:
            with (Path(TMP)/'runtime-import-audit.json').open('w',encoding='utf-8') as f:json.dump(result,f,indent=2)

    def test_real_Windows_read_lease_blocks_writes_to_only_a_new_fixture_copy(self):
        if os.name!='nt':self.skipTest('Windows deny-write/delete handle fixture')
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root=Path(d);p=root/overlay.PIN['filename'];p.write_bytes(overlay.archive_path().read_bytes())
            before=c.digest(p)
            with c.v2.CacheReadLease(dict(cache_root=str(root),packages=[],sources=[])) as lease,patch.object(overlay,'archive_path',return_value=p):
                overlay.hold_read_only(lease)
                with self.assertRaises(PermissionError):
                    with p.open('r+b'):pass
            self.assertEqual(c.digest(p),before)

    def test_unchanged_V2_closure_and_operator_policy_do_not_include_the_overlay(self):
        cfg=c.lock();self.assertEqual(len(cfg['packages']),13);self.assertNotIn('packaging',{p['name'] for p in cfg['packages']})
        self.assertEqual(cfg['guest_root'],'/home/modly/pixal3d-native-proof-v2')
        self.assertEqual(cfg['operator_policy']['prelaunch_bytes'],21687472128)
        self.assertEqual(cfg['guards']['admission_RAM_bytes'],22*1024**3)
        self.assertEqual(cfg['guards']['live_host_RAM_floor_bytes'],12*1024**3)

if __name__=='__main__':unittest.main()
