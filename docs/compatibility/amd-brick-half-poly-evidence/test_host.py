"""Pure/static host safety checks; no worker, input loader or Job is executed."""
import sys
sys.dont_write_bytecode = True
import ast
from pathlib import Path
import unittest

E = Path(__file__).resolve().parent


class HostContractTests(unittest.TestCase):
    def source(self):
        path = E/'run_host.py'
        self.assertTrue(path.is_file(), 'The bounded half-poly stage host is missing')
        return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def contract(self):
        names = {'PYTHON', 'NODE', 'BLENDER', 'CAP', 'DEADLINE', 'FOLDERS', 'PREDECESSORS'}
        nodes = [n for n in self.source().body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
                 or (isinstance(n, ast.FunctionDef) and n.name in {'worker_command', 'accepted_worker_result'})]
        ns = {'Path': Path, 'E': E}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pure-host-contract>', 'exec'), ns)
        return ns

    def test_python_stages_use_isolated_bytecode_free_exact_workers(self):
        h = self.contract()
        for stage, script in [('baseline', 'baseline.py'), ('assess', 'assess.py'), ('inspect', 'inspect_worker.py')]:
            self.assertEqual(h['worker_command'](stage, Path('private-output')),
                             ['D:/AI/Trellis2AMD/venv/Scripts/python.exe', '-I', '-B', str(E/script)])
        with self.assertRaises(ValueError):
            h['worker_command']('unapproved', Path('private-output'))

    def test_reduce_uses_bundled_node_and_bounded_heap(self):
        self.assertEqual(self.contract()['worker_command']('reduce', Path('private-output')),
                         [str(Path('C:/Users/wattz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')),
                          '--max-old-space-size=1536', '--wasm-max-mem-pages=32768', str(E/'reduce.cjs')])

    def test_render_has_factory_startup_no_autoexec_and_explicit_output(self):
        self.assertEqual(self.contract()['worker_command']('render', Path('private-output')),
                         [str(Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')), '--background',
                          '--factory-startup', '--disable-autoexec', '--threads', '4', '--python-exit-code',
                          '1', '--python', str(E/'render_worker.py'), '--', '--output', str(Path('private-output'))])

    def test_stage_chain_requires_quality_pass_before_render(self):
        h = self.contract()
        self.assertEqual(h['FOLDERS'], {x: x for x in ['baseline', 'reduce', 'assess', 'render', 'inspect']})
        self.assertEqual(h['PREDECESSORS'], {'baseline': (), 'reduce': ('baseline',),
                                           'assess': ('baseline', 'reduce'),
                                           'render': ('baseline', 'reduce', 'assess'),
                                           'inspect': ('baseline', 'reduce', 'assess', 'render')})
        for stage in h['FOLDERS']:
            for result in ['PASS', 'NO-GO', 'BLOCKED', 'FAIL', 'UNKNOWN']:
                self.assertEqual(h['accepted_worker_result'](stage, result),
                                 result == 'PASS' or (stage == 'assess' and result in {'NO-GO', 'BLOCKED'}))

    def test_process_and_job_memory_caps_and_deadline_remain_fixed(self):
        h = self.contract()
        self.assertEqual((h['CAP'], h['DEADLINE']), (6*1024**3, 180))
        assignments = [n for n in ast.walk(self.source()) if isinstance(n, ast.Assign)]
        self.assertTrue(any({ast.unparse(t) for t in n.targets} == {'limits.ProcessMemory', 'limits.JobMemory'}
                            and ast.unparse(n.value) == 'CAP' for n in assignments))

    def test_authentication_and_resource_admission_precede_exclusive_attempt_and_go(self):
        tree = self.source()
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        line = lambda text: min(n.lineno for n in calls if ast.unparse(n.func) == text)
        marker = min(n.lineno for n in calls if ast.unparse(n.func) == 'save'
                     and n.args and ast.unparse(n.args[0]) == 'marker')
        resources = sorted(n.lineno for n in calls if ast.unparse(n.func) == 'helper.resources')
        self.assertLess(resources[0], line('authenticate'))
        self.assertLess(line('authenticate'), resources[1])
        self.assertLess(resources[1], line('out.mkdir'))
        self.assertLess(line('out.mkdir'), marker)
        self.assertLess(marker, line('subprocess.Popen'))
        self.assertLess(line('subprocess.Popen'), line('job.attach'))
        self.assertLess(line('job.attach'), line('child.stdin.write'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        preflight = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == 'preflight')
        self.assertTrue(any(isinstance(n, ast.Return) for n in preflight.body))
        self.assertLess(preflight.lineno, marker)

    def test_receipt_identity_and_manifest_are_required_without_automatic_stage_launch(self):
        tree = self.source()
        text = ast.unparse(tree)
        self.assertIn("set(r['candidate_source_SHA256'])", text)
        self.assertIn("prior_host['worker_receipt_sha256'] == digest(result)", text)
        self.assertIn("receipt['classification'] == 'PASS'", text)
        self.assertIn("record['worker_receipt_sha256'] = digest(result)", text)
        self.assertIn("cleanup.get('owned_tree_exit_verified')", text)
        self.assertIn("not cleanup.get('forced_job_cleanup', True)", text)
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'subprocess.Popen'
                             for n in ast.walk(tree)), 1)


if __name__ == '__main__':
    unittest.main()
