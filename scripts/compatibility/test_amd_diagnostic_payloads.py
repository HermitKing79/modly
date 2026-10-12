"""CPU-only replay of the actual diagnostic expressions; no AI runtime imports."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import amd_forward_diagnostics as diagnostics

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / 'docs/compatibility/amd-instrumented-forward-evidence/run'
SAVED = json.loads((HISTORY / 'results.json').read_text())
GPU = SAVED['worker']['gpu']
RUNTIME = SAVED['worker']['runtime']
HISTORICAL_TRACE = [json.loads(line) for line in (HISTORY / 'trace.jsonl').read_text().splitlines()]
PROCESS = next(row for row in HISTORICAL_TRACE if row['event'] == 'process')
CORE = ('event', 'name', 'origin', 'seq', 'elapsed_seconds', 'monotonic_seconds', 'timestamp_utc')
CALL_FIELDS = ('duration_seconds', 'completion', 'error_type', 'error')
MEMORY = {'allocated_bytes': 0, 'reserved_bytes': 0, 'peak_allocated_bytes': 0,
          'peak_reserved_bytes': 0, 'available_bytes': 16000000000,
          'total_bytes': 17095983104, 'sampled_global_used_bytes': 1095983104}
INPUT = {'shape': [1, 8, 1, 1, 1], 'dtype': 'torch.float32', 'device': 'cuda:0',
         'stride': [8, 1, 1, 1, 1], 'layout': 'torch.strided',
         'contiguous': True, 'requires_grad': False}
REPLAY_ROWS = []
REPLAY_SHUTDOWN = {}


def production_calls(filename):
    tree = ast.parse((ROOT / 'scripts/compatibility' / filename).read_text())
    return sorted([node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == 'trace' and node.func.attr in ('emit', 'call')],
                  key=lambda node: node.lineno)


def execute(node, context):
    """Execute the unchanged production logging expression, never its worker."""
    return eval(compile(ast.Expression(node), '<production-diagnostic-expression>', 'eval'), context)


class PayloadTests(unittest.TestCase):
    def test_complete_saved_gpu_metadata_survives_actual_worker_call_and_journal(self):
        self.assertEqual(GPU, {'name': 'AMD Radeon RX 9070 XT', 'architecture': 'gfx1201',
                               'total_bytes': 17095983104})
        node = next(node for node in production_calls('amd_sparse_decoder_forward.py')
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == 'gpu')
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            with diagnostics.Journal(Path(tmp) / 'trace.jsonl') as journal:
                trace = diagnostics.Trace(journal.consume)
                execute(node, {'trace': trace, 'record': {'gpu': copy.deepcopy(GPU)}})
                self.assertEqual(journal.last['name'], 'gpu')
                self.assertEqual(journal.last['gpu'], GPU)
                self.assertEqual(json.loads((Path(tmp) / 'trace.jsonl').read_text())['gpu'], GPU)

    def test_reserved_record_and_argument_keys_fail_before_send_or_sequence_change(self):
        for key in CORE + ('self', 'call'):
            with self.subTest(key=key):
                rows = []; trace = diagnostics.Trace(rows.append)
                with self.assertRaisesRegex((ValueError, TypeError), 'reserved|Reserved|conflict'):
                    trace.emit('state', 'identity', **{key: 'overwritten'})
                self.assertEqual(rows, [])
                self.assertEqual(trace.seq, 0)

    def test_call_rejects_reserved_metadata_before_wrapped_operation(self):
        for key in CORE + CALL_FIELDS + ('self', 'call'):
            with self.subTest(key=key):
                rows = []; operations = []; trace = diagnostics.Trace(rows.append)
                with self.assertRaisesRegex((ValueError, TypeError), 'reserved|Reserved|conflict'):
                    trace.call('aten:convolution', lambda: operations.append(1), **{key: 'overwritten'})
                self.assertEqual(operations, [])
                self.assertEqual(rows, [])

    def test_reserved_names_remain_valid_inside_nested_metadata(self):
        nested = {key: 'metadata-value' for key in CORE + CALL_FIELDS + ('self', 'call')}
        rows = []; diagnostics.Trace(rows.append).emit('state', 'identity', identity=nested)
        parsed = json.loads(rows[0][len(diagnostics.MARKER):])
        self.assertEqual(parsed['name'], 'identity')
        self.assertEqual(parsed['seq'], 1)
        self.assertEqual(parsed['identity'], nested)

    def test_runtime_version_string_subclass_matches_installed_torch_representation(self):
        # torch.__version__ is TorchVersion(str), not an exact built-in str.
        # Inspect its declaration, never import torch or execute that module.
        source = Path('D:/AI/Trellis2AMD/venv/Lib/site-packages/torch/torch_version.py')
        tree = ast.parse(source.read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                          and node.name == 'TorchVersion')
        self.assertEqual([node.id for node in definition.bases], ['str'])
        class VersionString(str): pass
        runtime = {**RUNTIME, 'torch': VersionString('2.13.0+rocm10.0.0')}
        node = next(node for node in production_calls('amd_sparse_decoder_forward.py')
                    if len(node.args)>1 and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == 'runtime')
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            with diagnostics.Journal(Path(tmp) / 'trace.jsonl') as journal:
                execute(node, {'trace': diagnostics.Trace(journal.consume), 'record': {'runtime': runtime}})
                self.assertEqual(journal.last['torch'], '2.13.0+rocm10.0.0')

    def test_malformed_metadata_never_emits_or_spends_trace_budget(self):
        for payload in ({'gpu': {'total_bytes': float('nan')}}, {'x': float('inf')},
                        {'gpu': {1: 'non-string-key'}}, {'gpu': object()},
                        {'gpu': b'bytes'}, {'x': 'x' * 17000}):
            with self.subTest(payload=repr(payload)[:100]):
                rows = []; trace = diagnostics.Trace(rows.append)
                with self.assertRaisesRegex((ValueError, TypeError), 'diagnostic|Diagnostic'):
                    trace.emit('state', 'gpu', **payload)
                self.assertEqual(rows, [])
                self.assertEqual(trace.seq, 0)

    def test_journal_rejects_duplicate_fields_overflow_and_malformed_clock_identity(self):
        lines = []; diagnostics.Trace(lines.append).emit('state', 'gpu', gpu=GPU)
        valid = json.loads(lines[0][len(diagnostics.MARKER):])
        duplicate = lines[0][:-1] + ', "seq": 7}'
        overflow = lines[0][:-1] + ', "gpu_extra": 1e400}'
        bad_rows = [{**valid, 'monotonic_seconds': True}, {**valid, 'timestamp_utc': None},
                    {**valid, 'origin': 'arbitrary'}, {**valid, 'elapsed_seconds': -1},
                    {key: value for key, value in valid.items() if key != 'monotonic_seconds'}]
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            with diagnostics.Journal(Path(tmp) / 'trace.jsonl') as journal:
                for line in [duplicate, overflow] + [diagnostics.MARKER + json.dumps(row) for row in bad_rows]:
                    with self.subTest(line=line):
                        with self.assertRaises(ValueError): journal.consume(line)
                self.assertEqual(journal.rows, 0)
                self.assertIsNone(journal.last)

    def test_actual_diagnostic_call_sites_replay_lifecycle_and_shutdown(self):
        """Mutation of a production payload fails here, not only in a copied call."""
        replay = []; operations = []
        record = {'runtime': copy.deepcopy(RUNTIME), 'gpu': copy.deepcopy(GPU),
                  'before_placement': MEMORY.copy(), 'before_forward': MEMORY.copy(),
                  'after_forward': {**MEMORY, 'allocated_bytes': 256},
                  'release': {**MEMORY, 'model_collected': True, 'input_collected': True,
                              'output_collected': True, 'cpu_memory': SAVED['worker']['cpu_load']['memory_after']}}
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            path = Path(tmp) / 'trace.jsonl'; journal = diagnostics.Journal(path)
            def send(line): replay.append(journal.consume(line))
            trace = diagnostics.Trace(send)
            instance = object(); tensor = object()
            def describe(value):
                self.assertIs(value, tensor); return INPUT.copy()
            def original_call(value, *args, **kwargs):
                self.assertIs(value, instance); operations.append('module'); return 'module-result'
            def convolution(*args, **kwargs): operations.append('convolution'); return 'convolution-result'
            def synchronize(device):
                self.assertEqual(device, 0); operations.append('synthetic-sync')
            context = {'trace': trace, 'record': record, 'name': 'single-forward', 'phase': 'forward',
                       'permit': SimpleNamespace(calls=1), 'root_calls': 1, 'instance': instance,
                       'original_call': original_call, 'args': (tensor,), 'kwargs': {},
                       'detail': INPUT.copy(), 'tensor_description': describe, 'func': convolution,
                       'torch': SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))}
            try:
                nodes = production_calls('amd_sparse_decoder_forward.py')
                # Actual logging expressions, ordered as a representative lifecycle
                # rather than the lexical order of nested worker function definitions.
                for index in (1, 4, 5, 6, 0, 2, 3, 7, 8): execute(nodes[index], context)
                activity = {key: value for key, value in PROCESS.items() if key not in CORE}
                activity_node = production_calls('amd_forward_diagnostics.py')[0]
                execute(activity_node, {'trace': diagnostics.Trace(send, origin='supervisor'),
                                        'activity': SimpleNamespace(sample=lambda: copy.deepcopy(activity))})
                windows_node = production_calls('amd_forward_windows_monitor.py')[0]
                capture = json.loads((HISTORY / 'windows-after-exit.json').read_text())
                execute(windows_node, {'trace': diagnostics.Trace(send, origin='supervisor'),
                                       'good': True, 'duration': .1, 'capture': capture})
                execute(nodes[9], context)
                context['name']='complete';context['phase']='release'
                execute(nodes[0], context)
            finally:
                # Production termination records close/error fields rather than a
                # fabricated worker-terminated Trace event. Replay those exact
                # cleanup assignments with this real CPU journal.
                tree = ast.parse((ROOT / 'scripts/compatibility/amd_sparse_decoder_forward.py').read_text())
                shutdown = {}
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Assign) and len(node.targets)==1
                            and isinstance(node.targets[0], ast.Subscript)
                            and isinstance(node.targets[0].value, ast.Name)
                            and node.targets[0].value.id=='process'
                            and isinstance(node.targets[0].slice, ast.Constant)
                            and node.targets[0].slice.value in ('diagnostic_journal_stopped','diagnostic_journal_error')):
                        exec(compile(ast.Module(body=[node],type_ignores=[]), '<production-shutdown>', 'exec'),
                             {'journal':journal,'process':shutdown})
                self.assertEqual(shutdown, {'diagnostic_journal_stopped':True,'diagnostic_journal_error':None})
                REPLAY_SHUTDOWN.update(shutdown)
            self.assertFalse(journal.thread.is_alive())
            self.assertIsNone(journal.error)
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(saved, replay)
            self.assertEqual(len(nodes), 10)
            self.assertEqual(operations, ['module', 'convolution', 'synthetic-sync'])
            self.assertEqual(next(row for row in saved if row['name'] == 'gpu')['gpu'], GPU)
            self.assertEqual(next(row for row in saved if row['name'] == 'runtime')['hip'], '7.15.26333')
            self.assertEqual([row['completion'] for row in saved if row['event'] == 'return'],
                             ['host-return-only'] * 3)
            self.assertEqual(next(row for row in saved if row['name'] == 'owned-tree')['processes'], PROCESS['processes'])
            self.assertEqual(saved[-1]['name'], 'complete')
            REPLAY_ROWS[:] = saved

    def test_cpu_replay_has_not_imported_any_ai_runtime(self):
        self.assertFalse(any(name.split('.')[0] in ('torch', 'triton', 'trellis2', 'safetensors')
                             for name in sys.modules))


if __name__ == '__main__': unittest.main()
