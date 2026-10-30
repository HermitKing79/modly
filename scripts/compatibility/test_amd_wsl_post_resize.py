"""Offline fixtures; no WSL, real block devices, VHDX or E: access."""
import ast,contextlib,io,json,os,stat,struct,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
try:
    import amd_wsl_post_resize_geometry as g
    import amd_wsl_post_resize as h
except ModuleNotFoundError:g=h=None

class Implementation(unittest.TestCase):
    def test_implemented(self):self.assertIsNotNone(g);self.assertIsNotNone(h)

@unittest.skipIf(g is None,'Implementation missing')
class Geometry(unittest.TestCase):
    def header(self,capacity=64*1024**3,checksummed=True):
        data=bytearray(1024);struct.pack_into('<I',data,4,capacity//4096);struct.pack_into('<I',data,24,2)
        struct.pack_into('<H',data,56,0xef53)
        if checksummed:
            struct.pack_into('<I',data,100,0x400);data[0x175]=1
            struct.pack_into('<I',data,1020,g.crc32c(data[:1020]))
        return data
    def row(self):return dict(ext4_rw=True,root_device_bytes=64*1024**3,usable_bytes=63*1024**3,
        available_bytes=30*1024**3,fragment_bytes=4096,block_size_bytes=4096,total_blocks=63*1024**3//4096,
        free_blocks=31*1024**3//4096,available_blocks=30*1024**3//4096,
        superblock=dict(status='UNAVAILABLE',reason='PermissionError'))
    def test_crc_known_castagnoli_vector(self):self.assertEqual(g.crc32c(b'123456789')^0xffffffff,0xe3069283)
    def test_exact64_and_metadata_checksum(self):
        result=g.parse_superblock(self.header(),64*1024**3)
        self.assertEqual(result['filesystem_bytes'],64*1024**3);self.assertTrue(result['checksum_verified'])
    def test_corrupt_truncated_oversized_superblocks_rejected(self):
        changed=self.header();changed[4]^=1
        bad_magic=self.header(checksummed=False);bad_magic[56]=0
        for data in (changed,bad_magic,b'bad',self.header(capacity=128*1024**3)):
            with self.assertRaises(ValueError):g.parse_superblock(data,64*1024**3)
    def test_high_count_used_only_with_64bit_feature(self):
        data=self.header(checksummed=False);struct.pack_into('<I',data,0x150,1)
        with self.assertRaises(ValueError):g.parse_superblock(data,64*1024**3)
    def test_usable_size_never_claims_exact_geometry(self):
        result=g.assess(self.row())
        self.assertEqual(result['expansion'],'EXPANDED_BEYOND_32_GIB_EXACT_UNAVAILABLE')
        self.assertEqual(result['decoder_disk_gate'],'PASS');self.assertFalse(result['exact_64GiB_confirmed'])
    def test_full_partial_unchanged_and_low_free_classified(self):
        for size,label in ((64,'FULL_64_GIB_CONFIRMED'),(48,'PARTIAL_EXPANSION_CONFIRMED'),(32,'UNCHANGED_32_GIB')):
            row=self.row();row['usable_bytes']=31*1024**3;row['total_blocks']=row['usable_bytes']//4096
            row['available_bytes']=6*1024**3;row['available_blocks']=row['available_bytes']//4096
            row['free_blocks']=7*1024**3//4096
            row['superblock']=g.parse_superblock(self.header(capacity=size*1024**3),64*1024**3)
            self.assertEqual(g.assess(row)['expansion'],label)
        row=self.row();row['available_bytes']=5*1024**3-4096;row['available_blocks']=row['available_bytes']//4096
        self.assertEqual(g.assess(row)['decoder_disk_gate'],'FAIL')
    def test_bad_mount_or_inconsistent_counts_no_go(self):
        for change in (dict(ext4_rw=False),dict(root_device_bytes=32*1024**3),dict(available_bytes=99),dict(free_blocks=0)):
            row=self.row();row.update(change)
            with self.assertRaises(ValueError):g.assess(row)
    def test_permission_denial_is_explicit_no_escalation(self):
        with patch.object(g.os,'open',side_effect=PermissionError(13,'denied')):
            row=g.read_superblock('/dev/sdb',42,64*1024**3)
        self.assertEqual(row['status'],'UNAVAILABLE');self.assertEqual(row['errno'],13)
    def test_device_open_is_readonly_nofollow_and_identity_checked(self):
        with patch.object(g.os,'open',return_value=7) as opened,patch.object(g.os,'close'),patch.object(g.os,'fstat',return_value=SimpleNamespace(st_mode=0,st_rdev=42)):
            with self.assertRaises(ValueError):g.read_superblock('/dev/sdb',42,64*1024**3)
        self.assertFalse(opened.call_args.args[1]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
    def test_exact_device_reads_are_bounded_and_repeated(self):
        info=SimpleNamespace(st_mode=stat.S_IFBLK,st_rdev=42,st_dev=1,st_ino=2)
        with patch.object(g.os,'open',return_value=7),patch.object(g.os,'close') as close,patch.object(g.os,'fstat',return_value=info),patch.object(g.os,'pread',return_value=self.header(),create=True) as read:
            result=g.read_superblock('/dev/sdb',42,64*1024**3)
        self.assertEqual(result['filesystem_bytes'],64*1024**3)
        self.assertEqual([c.args for c in read.call_args_list],[(7,1024,1024),(7,1024,1024)]);close.assert_called_once_with(7)
    def test_symlink_error_is_not_downgraded_to_unavailable(self):
        with patch.object(g.os,'open',side_effect=OSError(40,'symlink loop')):
            with self.assertRaises(OSError):g.read_superblock('/dev/sdb',42,64*1024**3)
    def test_existing_audit_policy_and_mount_logic_reused(self):
        old=ast.parse(h.o.GUEST.read_text());new=ast.parse(h.GUEST.read_text())
        for name in ('audit','read','start_ticks','root_writable'):
            find=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(find(old)),ast.dump(find(new)))

@unittest.skipIf(h is None,'Implementation missing')
class Host(unittest.TestCase):
    def test_full_run_is_capacity_only_and_stops_on_guards(self):
        row=Geometry().row()
        for failure in ('none','admission','query','backup-close','report'):
            with tempfile.TemporaryDirectory(dir=h.b.ROOT.parent/'artifacts') as d:
                path=Path(d);case=path/'case';auth=path/'auth.json';auth.write_text('{}')
                identity=lambda kind:dict(file_id=1 if kind=='source' else 2,links=1)
                metadata=lambda kind:dict(virtual_capacity_bytes=(64 if kind=='source' else 32)*1024**3)
                settled=dict(identity=identity('source'),metadata=metadata('source'))
                cold=dict(backup=dict(identity=identity('backup'),metadata=metadata('backup')))
                @contextlib.contextmanager
                def opened(p,purpose):
                    f=io.BytesIO(b'fixture');f.kind='backup' if p==h.o.BACKUP else 'source'
                    try:yield f
                    finally:
                        f.close()
                        if f.kind=='backup' and failure=='backup-close':raise OSError('close failed')
                ob=h.o.Observer()
                def check(*a):
                    if failure=='admission':raise ValueError('below22')
                    return dict(available_bytes=24*1024**3)
                plain=lambda p,*a:SimpleNamespace(stat=lambda:SimpleNamespace(st_ino=1,st_nlink=1,st_size=100,st_mtime_ns=1)) if p==h.b.SOURCE else p
                original=Path.open
                def file_open(p,*a,**kw):
                    if failure=='report' and p.name=='REPORT.txt':raise OSError('report failed')
                    return original(p,*a,**kw)
                with contextlib.ExitStack() as stack:
                    for target,name,value in [(h,'CASE',case),(h,'AUTH',auth),(h,'publication',lambda c:None),(h,'authenticate',lambda a:(settled,cold)),(h,'final_inventory',lambda *a:None),(h.o,'no_inherited_job',lambda:None),(h.o,'inventory',lambda *a:None),(h.o,'Observer',lambda:ob),(h.b,'authenticate_worker_python',lambda:None),(h.b,'digest',lambda p:h.AUTH_SHA),(h.b,'authority',lambda:{}),(h.b,'inputs',lambda a:{}),(h.b,'Pins',lambda:SimpleNamespace(add=lambda *a:None,close=lambda:None)),(h.b,'native_file',opened),(h.b,'file_identity',lambda f:identity(f.kind)),(h.b,'metadata',lambda f:metadata(f.kind)),(h.b,'plain',plain),(h.r,'verify_backup',lambda *a:None),(ob,'check',check),(ob,'preflight',lambda:None),(h.time,'sleep',lambda s:None),(Path,'open',file_open)]:stack.enter_context(patch.object(target,name,value))
                    query=stack.enter_context(patch.object(h,'query',side_effect=ValueError('query failed') if failure=='query' else None,return_value=row))
                    stack.enter_context(patch.object(h.o,'invoke_native',side_effect=AssertionError('resize forbidden')))
                    # Receipt loader needs only these three accepted files; all are C: fixtures.
                    accepted=path/'accepted';accepted.mkdir()
                    for name in ('final-result.json','worker-result.json','destination-hash-stdout.txt','inventory-process.json','backup-worker-process.json','final-inventory-process.json'):(accepted/name).write_text('{}')
                    stack.enter_context(patch.object(h.o,'ACCEPTED',accepted))
                    code=h.run('a'*40)
                self.assertEqual(code,0 if failure=='none' else 1)
                if failure=='admission':query.assert_not_called()
                if failure=='report':self.assertFalse((case/'final-result.json').exists())
                else:
                    result=json.loads((case/'final-result.json').read_text());self.assertEqual(result['classification'],'PASS_SPACE_GATE' if failure=='none' else 'FAIL')
    def test_inert_default_and_approval(self):
        with patch.object(h,'run',side_effect=AssertionError('live')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h.main([]),0)
            with self.assertRaises(SystemExit):h.main(['--run'])
    def test_capacity_route_only_and_global_bindings_restored(self):
        original=(h.o.CASE,h.o.GUEST)
        with patch.object(h.o,'capacity_query',return_value={'fixture':1}) as query,patch.object(h.o,'invoke_native',side_effect=AssertionError('resize forbidden')):
            self.assertEqual(h.query(SimpleNamespace(check=lambda:None)),{'fixture':1})
        self.assertEqual((h.o.CASE,h.o.GUEST),original);query.assert_called_once()
    def test_live_boundary_requires_final22(self):
        ob=SimpleNamespace(check=lambda:(_ for _ in ()).throw(ValueError('below22')))
        with patch.object(h.o,'capacity_query',side_effect=AssertionError('no dispatch')):
            with self.assertRaisesRegex(ValueError,'below22'):h.query(ob)
    def test_saved_native_message_cannot_prove_block_geometry(self):
        stdout='The operation completed successfully. \r\n'.encode('utf-16-le')
        stderr=b'e2fsck 1.47.0 (5-Feb-2023)\nPass 1: Checking inodes, blocks, and sizes\nresize2fs 1.47.0 (5-Feb-2023)\n'
        text='\n'.join(x.replace(b'\0',b'').decode('utf-8') for x in (stdout,stderr))
        with self.assertRaisesRegex(ValueError,'geometry unavailable'):h.o.filesystem_bytes(text)

if __name__=='__main__':unittest.main(verbosity=2)
