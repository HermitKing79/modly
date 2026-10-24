"""Windows audit-path regression tests. Never run project preparation or query RAM."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name('amd_zombie_ram_diagnostic.py')
BASE = '1553e73a7bf27ff2de0f331861595e02b0b9a30e'
SCRATCH = ROOT.parent/'artifacts/amd-zombie-windows-audit-tests'
SCRATCH.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('zombie_windows_audit_target', SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@unittest.skipUnless(sys.platform == 'win32', 'Requires CPython Windows subprocess audit events')
class WindowsAudit(unittest.TestCase):
    def probe(self, name):
        result = subprocess.run([sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve()), '--probe', name],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_legitimate_bootstrap_uses_actual_windows_audit_event(self):
        result = self.probe('implicit-rev-parse')
        self.assertEqual(result['status'], 'allowed', result)
        self.assertEqual(result['events'][0], dict(executable=None, args_type='str',
            command_line='git rev-parse HEAD', correct_cwd=True))
        self.assertRegex(result['stdout'].strip(), r'^[0-9a-f]{40}$')
        self.assertEqual(result['memory_checks'], 1)

    def test_authorised_git_show_uses_actual_windows_audit_event(self):
        result = self.probe('implicit-show')
        self.assertEqual(result['status'], 'allowed', result)
        self.assertEqual(result['events'][0]['command_line'], 'git show '+BASE+':README.md')
        expected = hashlib.sha256((ROOT/'README.md').read_text(encoding='utf-8').encode('utf-8')).hexdigest()
        self.assertEqual(result['stdout_sha256'], expected)
        self.assertEqual(result['memory_checks'], 1)

    def test_explicit_pinned_executable_and_quoted_absolute_argv0(self):
        for case in ('explicit-executable', 'absolute-argv0', 'git-exe-alias'):
            with self.subTest(case=case):
                result = self.probe(case)
                self.assertEqual(result['status'], 'allowed', result)
                self.assertEqual(result['events'][0]['args_type'], 'str')
                self.assertRegex(result['stdout'].strip(), r'^[0-9a-f]{40}$')
                self.assertEqual(result['memory_checks'], 1)

    def test_mutation_injection_shell_foreign_executable_and_cwd_are_rejected(self):
        for case in ('mutation', 'injection-list', 'injection-string', 'quote-variant', 'extra-option',
                     'newline', 'wsl', 'foreign-executable', 'foreign-git', 'foreign-argv0', 'wrong-cwd', 'shell',
                     'shell-with-git', 'traversal', 'double-space', 'leading-space'):
            with self.subTest(case=case):
                result = self.probe(case)
                self.assertEqual(result['status'], 'blocked', result)
                self.assertEqual(result['events'][0]['args_type'], 'str', result)
                self.assertEqual(result['memory_checks'], 0, result)


def probe(name):
    evidence = Path(tempfile.mkdtemp(dir=SCRATCH))
    git = str(Path(shutil.which('git')).resolve())
    command, kwargs = ['git', 'rev-parse', 'HEAD'], dict(cwd=str(ROOT))
    if name == 'implicit-show': command = ['git', 'show', BASE+':README.md']
    elif name == 'explicit-executable': kwargs['executable'] = git
    elif name == 'absolute-argv0': command[0] = git
    elif name == 'git-exe-alias': command[0] = 'git.exe'
    elif name == 'mutation': command = ['git', 'checkout', 'main']
    elif name == 'injection-list': command[-1] = 'HEAD;whoami'
    elif name == 'injection-string': command = 'git rev-parse HEAD & whoami'
    elif name == 'quote-variant': command = '"git" rev-parse HEAD'
    elif name == 'extra-option': command = 'git show '+BASE+':README.md --output=must-not-exist'
    elif name == 'newline': command = 'git rev-parse HEAD\nwhoami'
    elif name == 'wsl': command = ['wsl.exe']
    elif name == 'foreign-executable': kwargs['executable'] = sys.executable
    elif name == 'foreign-git':
        kwargs['executable'] = str(evidence/'git.exe')
        command[0] = str(evidence/'git.exe')
    elif name == 'foreign-argv0':
        kwargs['executable'] = git
        command[0] = str(evidence/'git.exe')
    elif name == 'wrong-cwd': kwargs['cwd'] = str(evidence)
    elif name == 'shell': kwargs['shell'] = True
    elif name == 'shell-with-git': kwargs.update(shell=True, executable=git)
    elif name == 'traversal': command = ['git', 'show', BASE+':../README.md']
    elif name == 'double-space': command = 'git  rev-parse HEAD'
    elif name == 'leading-space': command = ' git rev-parse HEAD'
    allowed = name in ('implicit-rev-parse', 'implicit-show', 'explicit-executable', 'absolute-argv0', 'git-exe-alias')
    events, checks = [], []
    def capture(event, args):
        if event == 'subprocess.Popen':
            events.append(dict(executable=args[0], args_type=type(args[1]).__name__, command_line=args[1],
                               correct_cwd=Path(args[2] or os.getcwd()).resolve() == ROOT))
    def check_memory(): checks.append('synthetic-no-RAM-query')
    sys.addaudithook(capture)
    diagnostic.install_guards(evidence, ROOT, check_memory)
    result = dict(status='unexpected', events=events)
    try:
        # Negative probes retain all actual CPython audit formatting, but a
        # mocked final OS boundary makes mutations/WSL impossible even on failure.
        if allowed:
            stdout = subprocess.check_output(command, text=True, encoding='utf-8', timeout=15, **kwargs)
        else:
            with mock.patch('subprocess._winapi.CreateProcess', side_effect=AssertionError('Unsafe OS boundary reached')):
                stdout = subprocess.check_output(command, text=True, encoding='utf-8', timeout=15, **kwargs)
        result.update(status='allowed', stdout=None if name=='implicit-show' else stdout,
                      stdout_sha256=hashlib.sha256(stdout.encode('utf-8')).hexdigest())
    except diagnostic.DiagnosticStop as exc:
        result.update(status='blocked', error=str(exc))
    except BaseException as exc:
        result.update(status='unexpected-error', error=type(exc).__name__+': '+str(exc))
    result['memory_checks'] = len(checks)
    print(json.dumps(result), flush=True)
    os._exit(0)  # Permanent guard forbids deleting the retained synthetic directory.


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--probe': probe(sys.argv[2])
    else: unittest.main()
