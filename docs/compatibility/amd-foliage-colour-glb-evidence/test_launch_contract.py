"""Actual command/argument parser boundary, without launching Blender."""
import sys
sys.dont_write_bytecode=True
import importlib.util,subprocess,unittest
from pathlib import Path
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E))
import run_host,render_worker


class LaunchTests(unittest.TestCase):
    def command(self):
        self.assertTrue(callable(getattr(run_host,'render_command',None)),'Actual host command constructor required')
        self.assertTrue(callable(getattr(render_worker,'script_args',None)),'Actual guest argument parser required')
        return run_host.render_command(render_worker.OUT)

    def test_exact_production_command_selects_guest_output_after_separator(self):
        command=self.command()
        self.assertEqual(command[-3:],['--','--output',str(render_worker.OUT)])
        self.assertEqual(render_worker.script_args(command).output,render_worker.OUT)
        self.assertEqual(render_worker.OUT,run_host.P/'render-qualified')
        self.assertNotEqual(render_worker.OUT,run_host.P/'render')

    def test_old_missing_separator_reproduces_argument_rejection(self):
        command=self.command();command.remove('--')
        with self.assertRaises(SystemExit) as result:render_worker.script_args(command)
        self.assertEqual(result.exception.code,2)

    def test_real_corrected_script_route_reaches_strict_GO_without_blender(self):
        command=self.command()
        probe=subprocess.run([sys.executable,'-I','-B',str(E/'render_worker.py'),*command[1:]],input=b'NO\n',capture_output=True,timeout=8)
        self.assertEqual(probe.returncode,1)
        self.assertIn(b'FOLIAGE_COLOUR_GLB_READY_GO',probe.stdout)
        self.assertIn(b'Exact binary GO admission rejected',probe.stderr)
        self.assertNotIn(b'unrecognized arguments',probe.stderr)


if __name__=='__main__':unittest.main()
