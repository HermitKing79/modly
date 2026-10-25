"""Guarded CPU preparation child. Use Invoke-ZombieRamDiagnostic.ps1 after closing apps.

Importing this module is stdlib-only. No historical generation entrypoint is run.
The real worker requires a fresh launcher receipt and a phase-by-phase stdin permit.
"""
import argparse
import ast
import builtins
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time

BASE = '28f4f0163833d59d861404719e6b2cb042083f9e'
FLOOR = 22 * 1024**3
PREFERRED = 24 * 1024**3
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('amd_zombie_ram_sources.json')
CHECKPOINT = Path('D:/AI/Trellis2AMD/ComfyUI/models/microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors')
CPU_STEPS = ('imports', 'construct_runtime', 'validate_lineage', 'verify_sources',
             'stream_checkpoint', 'metadata_checks')
DENIED_CALLS = frozenset(('main', 'prepare', 'supervise', 'supervisor', 'reserve_attempt',
    'reserve_fresh', 'worker', 'worker_command', 'guest_bootstrap', 'load_cpu', 'load_gpu',
    '_factory', '_import_parent', 'forward', 'launch', 'run_worker'))


class DiagnosticStop(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def admitted(values):
    return len(values) == 3 and all(type(n) is int and n >= FLOOR for n in values)


def forbidden_import(name):
    return (name.split('.')[0] in {'torch', 'triton', 'trellis2', 'flex_gemm', 'transformers',
                                  'diffusers', 'timm', 'onnxruntime', 'comfy', 'comfyui'}
            or name == 'safetensors.torch' or name.startswith('safetensors.torch.'))


def forbidden_call(name, filename, root):
    if name not in DENIED_CALLS:
        return False
    try:
        Path(filename).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return name in DENIED_CALLS and Path(filename).resolve() != Path(__file__).resolve()


def validate_write(path, evidence):
    path = Path(os.fsdecode(path)).absolute()
    try:
        path.resolve().relative_to(evidence.resolve())
    except ValueError as exc:
        raise DiagnosticStop('Write outside unique diagnostic evidence denied: ' + str(path)) from exc
    for parent in (path, *path.parents):
        if parent.exists() and (parent.is_symlink() or parent.is_junction()):
            raise DiagnosticStop('Redirected evidence path denied: ' + str(parent))


def windows_implicit_git_identity(git):
    """Reject pre-PATH shadows of the pinned .exe; do not emulate a shell search."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    buffer = ctypes.create_unicode_buffer(32768)
    def query(function, *args):
        function.restype = ctypes.c_uint32
        count = function(*args, buffer, len(buffer))
        if not 0 < count < len(buffer):
            raise DiagnosticStop('Windows executable search directory query failed')
        return Path(buffer.value)
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.GetSystemDirectoryW.argtypes = kernel.GetWindowsDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    application = query(kernel.GetModuleFileNameW, None).parent
    system = query(kernel.GetSystemDirectoryW)
    windows = query(kernel.GetWindowsDirectoryW)
    # Conservatively reject *any* foreign candidate in the documented pre-PATH
    # search directories, even if an earlier directory already contains our pin.
    for directory in (application, Path.cwd(), system, windows/'System', windows):
        candidate = directory/'git.exe'
        if candidate.exists() and candidate.resolve() != git:
            raise DiagnosticStop('Foreign implicit Windows Git executable denied')
    if Path(shutil.which('git.exe') or '').resolve() != git:
        raise DiagnosticStop('Windows PATH Git executable differs from pin')


def validate_spawn(executable, args, cwd, root, *, git=None):
    git = Path(git or shutil.which('git.exe' if os.name == 'nt' else 'git') or '').resolve()
    windows_command = os.name == 'nt' and isinstance(args, str)
    if windows_command:
        # Windows audits Popen *after* list2cmdline, often with executable=None.
        # Recognise only canonical spellings of the authorised argv; never parse
        # an arbitrary command line, shell quoting, options or operators.
        from subprocess import list2cmdline
        for program in ('git', 'git.exe', str(git)):
            prefix = list2cmdline([program]) + ' '
            if args.startswith(prefix):
                tail = args[len(prefix):]
                if tail == 'rev-parse HEAD':
                    args = [program, 'rev-parse', 'HEAD']
                else:
                    match = re.fullmatch(r'show ([0-9a-f]{40}:[A-Za-z0-9_./-]+)', tail)
                    if match:
                        args = [program, 'show', match[1]]
                break
    if (not isinstance(args, (tuple, list)) or not args or not all(isinstance(a, str) for a in args)
            or Path(args[0]).name.lower() not in ('git', 'git.exe')
            or Path(cwd or os.getcwd()).resolve() != root.resolve()):
        raise DiagnosticStop('Only bounded read-only Git subprocesses are permitted')
    effective = args[0] if executable is None and os.name == 'nt' else executable
    if windows_command and executable is not None and not Path(executable).is_absolute():
        raise DiagnosticStop('Windows explicit executable must be the absolute Git pin')
    for program in (args[0], effective):
        lookup = 'git.exe' if os.name == 'nt' and program in ('git', 'git.exe') else program
        if not isinstance(program, str) or Path(shutil.which(lookup) or program).resolve() != git:
            raise DiagnosticStop('Foreign Git executable denied')
    if windows_command and executable is None and args[0] in ('git', 'git.exe'):
        windows_implicit_git_identity(git)
    if list(args[1:]) == ['rev-parse', 'HEAD']:
        return
    if len(args) == 3 and args[1] == 'show' and re.fullmatch(r'[0-9a-f]{40}:[A-Za-z0-9_./-]+', args[2]):
        name = args[2].split(':', 1)[1]
        if '..' not in name.split('/') and not name.startswith('/'):
            return
    raise DiagnosticStop('Git mutation or unbounded command denied')


class NeuralBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden_import(fullname):
            raise DiagnosticStop('Neural import denied: ' + fullname)
        return None


def install_guards(evidence, root, check_memory):
    """Defence in depth around the pinned, unchanged CPU readers; no sandbox claim."""
    if any(forbidden_import(name) for name in sys.modules):
        raise DiagnosticStop('Worker must start without neural modules')
    resolved_git = shutil.which('git.exe' if os.name == 'nt' else 'git')
    if not resolved_git:
        raise DiagnosticStop('Pinned Git executable unavailable')
    git = Path(resolved_git).resolve(strict=True)
    if not git.is_file() or os.name == 'nt' and git.name.lower() != 'git.exe':
        raise DiagnosticStop('Windows Git pin must be a real Git .exe')
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if forbidden_import(name):
            raise DiagnosticStop('Neural import denied: ' + name)
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    sys.meta_path.insert(0, NeuralBlocker())
    def audit(event, args):
        if event == 'import' and forbidden_import(str(args[0])):
            raise DiagnosticStop('Neural import denied: ' + str(args[0]))
        if event == 'subprocess.Popen':
            validate_spawn(args[0], args[1], args[2], root, git=git)
            check_memory()
        if event.startswith('socket.') or event in ('os.system', 'os.startfile', 'os.exec', 'os.posix_spawn'):
            raise DiagnosticStop('External action denied: ' + event)
        if event == 'open' and not isinstance(args[0], int):
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                validate_write(args[0], evidence)
        if event in ('os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.link', 'os.symlink', 'os.truncate'):
            raise DiagnosticStop('Filesystem mutation denied: ' + event)
    sys.addaudithook(audit)
    def profile(frame, event, arg):
        if event == 'call' and forbidden_call(frame.f_code.co_name, frame.f_code.co_filename, root):
            raise DiagnosticStop('Generation/model call denied: ' + frame.f_code.co_name)
    sys.setprofile(profile)


def windows_memory():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (n, ctypes.c_uint64) for n in ('total', 'available', 'page_total', 'page_available',
                                          'virtual_total', 'virtual_available', 'extended')]
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
            (n, ctypes.c_size_t) for n in ('peak_ws', 'ws', 'peak_paged', 'paged', 'peak_nonpaged',
                                         'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
    memory = Memory(); memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return dict(available_bytes=memory.available, total_physical_bytes=memory.total,
                working_set_bytes=counters.ws, private_commit_bytes=counters.private,
                os_peak_working_set_bytes=counters.peak_ws)


class Phases:
    def __init__(self, evidence, sample, handshake=None):
        self.evidence, self.sample = evidence, sample
        self.handshake = handshake or self.permit
        self.current = 'worker-bootstrap'
        self.records = []
        self.started = time.monotonic()
        self.stream = (evidence / 'phases.jsonl').open('x', encoding='utf-8', buffering=1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def guard(self, value=None):
        value = self.sample() if value is None else value
        if value['available_bytes'] < FLOOR:
            row = dict(value, timestamp_utc=utc(), phase=self.current, edge='gate-rejection')
            if self.records:
                row['available_change_since_last_boundary_bytes'] = value['available_bytes'] - self.records[-1]['available_bytes']
            self.records.append(row)
            self.stream.write(json.dumps(row) + '\n'); self.stream.flush(); os.fsync(self.stream.fileno())
            raise DiagnosticStop('Below 22 GiB during ' + self.current + ': ' + str(value['available_bytes']))
        if time.monotonic() - self.started > 480:
            raise DiagnosticStop('Original 480-second host deadline reached')
        return value

    @staticmethod
    def permit(event):
        print('MODLY_RAM_PHASE=' + json.dumps(event), flush=True)
        if sys.stdin.readline().strip() != 'CONTINUE':
            raise DiagnosticStop('Observer permit missing; no further preparation')

    def run(self, name, operation):
        previous = self.current
        self.current = name
        before = self.record(name, 'before')
        self.guard(before)
        self.handshake(before)
        self.guard()
        try:
            result = operation()
        except BaseException as exc:
            self.record(name, 'error', error=type(exc).__name__ + ': ' + str(exc))
            raise
        after = self.record(name, 'after', before=before)
        self.guard(after)
        self.handshake(after)
        self.current = previous
        return result

    def record(self, name, edge, before=None, **extra):
        row = dict(timestamp_utc=utc(), elapsed_seconds=time.monotonic() - self.started,
                   phase=name, edge=edge, **self.sample(), **extra)
        if before:
            for key in ('available_bytes', 'working_set_bytes', 'private_commit_bytes'):
                if key in row and key in before:
                    row[key.replace('_bytes', '_delta_bytes')] = row[key] - before[key]
        self.records.append(row)
        self.stream.write(json.dumps(row) + '\n'); self.stream.flush(); os.fsync(self.stream.fileno())
        return row


def extract_checkpoint(source, namespace):
    """Compile only checkpoint_record, never the decoder's module body or main."""
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint_record']
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise DiagnosticStop('Exact pure checkpoint verifier missing')
    code = ast.Module(body=nodes, type_ignores=[])
    exec(compile(code, '<extracted-checkpoint-verifier>', 'exec'), namespace)
    return namespace['checkpoint_record']


def stream_digest(path, guard):
    before = Path(path).stat()
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            guard()
            block = stream.read(1024**2)
            if not block:
                break
            digest.update(block)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise DiagnosticStop('File changed during read-only streaming')
    return digest.hexdigest()


class CPUPreparation:
    """CPU subset of historical construction/preflight; owns no generation Runtime."""
    def __init__(self, phases, commit):
        self.phases, self.commit = phases, commit

    def imports(self):
        sys.path.insert(0, str(ROOT / 'scripts/compatibility'))
        import amd_zombie_decode
        self.zombie = amd_zombie_decode

    def construct_runtime(self):
        # Historical Runtime.__init__ first authenticates this same complete lineage.
        self.bundle = self.phases.run('constructor-source-lineage', lambda: self.zombie.source_inputs(self.zombie.shape.EVIDENCE))
        self.contract_module = self.zombie.historical.load_module('_modly_ram_cpu_run_contract', self.zombie.historical.CODE / 'run_contract.py')
        root = PurePosixPath('/home/modly/run-zombie-ram-diagnostic-no-execution')
        guest_path = self.zombie.historical.guest_path
        self.contract = self.contract_module.parse(dict(schema=self.contract_module.SCHEMA,
            initial_count=self.bundle['count'], input_paths={k: guest_path(v['path']) for k,v in self.bundle['inputs'].items()},
            input_sha256={k:v['sha256'] for k,v in self.bundle['inputs'].items()}, source_job_sha256=self.bundle['source_job_sha256'],
            output_directory=str(root/'geometry'), scratch_directory=str(root/'scratch'), repo_root=guest_path(ROOT),
            maximum_final_rows=1842968, maximum_intermediate_rows=1000000, native_stage=self.contract_module.NATIVE_STAGE,
            large_convolution_tiling=True))

    def validate_lineage(self):
        fresh = self.zombie.source_inputs(self.bundle['source_job'].parents[1])
        if fresh['source_job_sha256'] != self.contract.source_job_sha256:
            raise DiagnosticStop('Source shape job changed')

    def verify_sources(self):
        def imports():
            import amd_sparse_decoder_forward
            return amd_sparse_decoder_forward
        verifier = self.phases.run('source-verifier-imports', imports)
        host = self.zombie.historical
        names = sorted(set(host.STATIC_FILES + list(verifier.COMMITTED_FILES) + list(self.bundle['required_files']) +
            [p.relative_to(ROOT).as_posix() for p in host.CODE.glob('*.py')] + [
                'scripts/compatibility/amd_zombie_decode.py', 'scripts/compatibility/amd_zombie_decode_resume.py',
                'scripts/compatibility/test_amd_zombie_decode.py', 'scripts/compatibility/test_amd_zombie_decode_resume.py',
                'docs/compatibility/amd-zombie-decode-evidence/cpu-admission-summary.json',
                'docs/compatibility/amd-zombie-decode-evidence/wsl_cpu_probe.py',
                'docs/compatibility/amd-zombie-ram-readiness-3-evidence/readiness-summary.json',
                'docs/compatibility/amd-zombie-ram-headroom-evidence/memory_readiness.py']))
        self.sources = verifier.verify_committed_files(self.commit, names, timeout_seconds=15)
        if host.digest(ROOT/'scripts/compatibility/amd_wsl_native_import_policy.py') != '8b9207a14757ccda197e11c054d694f9b4ff47006a1cccc46d71f1d494ab96cd':
            raise DiagnosticStop('Qualified native policy changed')
        preserved = host.read_json(host.LEGACY.parent/'preservation-before.json')
        for field, parent in [('unchanged_control_code', ROOT/'scripts/compatibility'),
                              ('loader_code', ROOT/'experimental/extensions/trellis2-amd')]:
            for name, sha in preserved[field].items():
                self.phases.guard()
                if host.digest(parent/name) != sha:
                    raise DiagnosticStop('Qualified control/loader changed: ' + name)

    def stream_checkpoint(self):
        host = self.zombie.historical
        function = extract_checkpoint(host.GUEST_ENTRY.read_text(encoding='utf-8'),
            dict(json=json, sys=sys, EV=host.LEGACY.parent, digest=lambda p: stream_digest(p, self.phases.guard)))
        self.checkpoint = function(CHECKPOINT)

    def metadata_checks(self):
        def imports():
            sys.path[:0] = [str(ROOT/'api'), str(ROOT/'experimental/extensions/trellis2-amd')]
            from shape_decoder_loader import ShapeDecoderComponent
            return ShapeDecoderComponent
        component_type = self.phases.run('metadata-library-imports', imports)
        component = self.phases.run('metadata-component-construction', lambda: component_type(
            str(CHECKPOINT.with_suffix('.json')), str(CHECKPOINT)))
        self.phases.run('metadata-source-verification', component._verify_source)
        if component.loaded or component.diagnostics['weights_loaded']:
            raise DiagnosticStop('Metadata preflight unexpectedly loaded a model')
        self.phases.run('metadata-component-unload', component.unload)
        accepted = self.zombie.historical.read_json(ROOT/'docs/compatibility/amd-wsl-packaging-decoder-loading-evidence/dependency-run/results.json')
        if (accepted['classification'] != 'QUALIFIED' or accepted['worker_returncode'] != 0
                or not accepted['observer_healthy'] or accepted['windows_job_cleanup']['forced_job_cleanup']):
            raise DiagnosticStop('Qualified dependency evidence missing')


def prepare(backend, phases):
    for name in CPU_STEPS:
        phases.run(name, getattr(backend, name))


def authenticate_base(phases):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['base_commit'] != BASE:
        raise DiagnosticStop('Wrong base manifest')
    for name, sha in manifest['source_sha256'].items():
        if stream_digest(ROOT / name, phases.guard) != sha:
            raise DiagnosticStop('Original source changed: ' + name)


def run_child(evidence, token):
    if os.name != 'nt' or ctypes.windll.shell32.IsUserAnAdmin():
        raise DiagnosticStop('Normal non-elevated Windows user required')
    evidence = evidence.resolve(strict=True)
    if evidence.parent != (ROOT.parent/'artifacts').resolve() or not evidence.name.startswith('amd-zombie-standalone-ram-'):
        raise DiagnosticStop('Unique private diagnostic directory required')
    receipt = json.loads((evidence/'admission.json').read_text(encoding='utf-8-sig'))
    if receipt['token'] != token or not admitted([r['available_bytes'] for r in receipt['readings']]):
        raise DiagnosticStop('Fresh three-reading launcher receipt required')
    stamps = [datetime.fromisoformat(r['timestamp_utc']) for r in receipt['readings']]
    if any((b-a).total_seconds() < 4 for a,b in zip(stamps, stamps[1:])) or not 0 <= (datetime.now(timezone.utc)-stamps[-1]).total_seconds() < 60:
        raise DiagnosticStop('Stale or incorrectly spaced admission')
    phases = Phases(evidence, windows_memory)
    result = dict(classification='FAILED', real_diagnostic=True, neural_attempt_reserved=False,
                  weights_loaded=False, WSL_launched=False, neural_forwards=0, base_commit=BASE)
    try:
        install_guards(evidence, ROOT, phases.guard)
        # -S prevents sitecustomize/.pth execution before the guards. Add only the
        # original installed CPU dependencies, without processing startup hooks.
        sys.path.append('D:/AI/Trellis2AMD/venv/Lib/site-packages')
        # Windows venv python.exe is a redirector; the owned interpreter has a
        # different PID. Bind external telemetry before the first phase permit.
        print('MODLY_RAM_IDENTITY=' + json.dumps(dict(pid=os.getpid(), image=sys._base_executable)), flush=True)
        phases.run('before-project-imports', lambda: None)
        phases.run('original-source-authentication', lambda: authenticate_base(phases))
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=15).strip()
        prepare(CPUPreparation(phases, commit), phases)
        result['classification'] = 'CPU_PREPARATION_COMPLETE'
        result['harness_commit'] = commit
        returncode = 0
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc), failed_phase=phases.current)
        returncode = 2
    finally:
        result.update(timestamp_utc=utc(), last_phase=phases.current, phase_records=len(phases.records),
            forbidden_modules_present=[n for n in sys.modules if forbidden_import(n)],
            interpretation='Host availability is system-wide; working sets share pages and private commit is not physical residency. Temporal association alone does not establish causality.')
        with (evidence/'child-result.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
        phases.stream.close()
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--launcher-token')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(real_diagnostic=False, base_commit=BASE, admission_bytes=FLOOR,
            preferred_bytes=PREFERRED, CPU_steps=CPU_STEPS, direct_generation_entrypoint=False)))
        return 0
    if not args.evidence or not args.launcher_token:
        parser.error('Use the PowerShell launcher; direct execution has no admission authority')
    return run_child(args.evidence, args.launcher_token)


if __name__ == '__main__':
    raise SystemExit(main())
