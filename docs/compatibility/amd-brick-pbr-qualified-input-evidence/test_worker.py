"""Stdlib composition tests; actual ownership/array acceptance runs separately."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE/'worker.py'
OLD_WORKER = ROOT/'docs/compatibility/amd-brick-pbr-gpu-attempt-evidence/worker.py'
OLD_TEST = ROOT/'scripts/compatibility/test_amd_brick_pbr_worker.py'
sys.dont_write_bytecode = True


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), vars(module))
    return module


class QualifiedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = source_module('qualified_worker_composition', SOURCE)

    def test_the_production_gpu_and_cpu_modes_use_the_same_input_function(self):
        tree = ast.parse(SOURCE.read_bytes())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in ('execute_admitted', 'cpu_input_acceptance'):
            calls = [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'prepare_inputs']
            self.assertEqual(len(calls), 1)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'inputs'
                             and node.func.attr == 'prepare' for node in ast.walk(tree)))
        cpu_call = next(node for node in ast.walk(functions['cpu_input_acceptance'])
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'prepare_inputs')
        self.assertEqual(ast.dump(cpu_call.args[0]), "Name(id='authenticated_profile', ctx=Load())")
        self.assertEqual(ast.dump(cpu_call.args[1]),
                         ast.dump(ast.parse('dict(candidate_identities=identities)', mode='eval').body))
        cpu_calls = [node for node in ast.walk(functions['cpu_input_acceptance'])
                     if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == 'verify_manifest'
                            for node in cpu_calls))
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'authenticate'
                            for node in cpu_calls))

    def test_compute_and_guards_are_unchanged_except_the_input_call(self):
        def body(path):
            tree = ast.parse(path.read_bytes())
            functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            return functions
        old, new = body(OLD_WORKER), body(SOURCE)
        for name in ('validate_stage_go', 'stage_ready', 'qualify_staged_native', 'guest_available',
                     'authenticate_plan', 'bootstrap', 'worker'):
            self.assertEqual(ast.dump(old[name], include_attributes=False),
                             ast.dump(new[name], include_attributes=False), name)
        for node in ast.walk(old['execute_admitted']):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'prepare'):
                node.value = ast.parse('prepare_inputs(profile, plan, record)', mode='eval').body
        self.assertEqual(ast.dump(old['execute_admitted'], include_attributes=False),
                         ast.dump(new['execute_admitted'], include_attributes=False))

    def test_route_passes_only_fixed_qualified_authority_and_returns_actual_result(self):
        w = self.worker
        profile = object()
        prepared = dict(profile=profile, metadata=dict(cpu_only=True, model_imported=False,
                                                       gpu_initialized=False))
        security = dict(all_passed=True, tests=[dict(name='composition-fixture', passed=True)])
        prepare = Mock(return_value=dict(prepared=prepared, acceptance_security_results=security))
        require_prepared = Mock(return_value=prepared)
        plan = dict(candidate_identities={w.INPUT_POLICY_PATH.relative_to(ROOT).as_posix(): w.INPUT_POLICY_SHA256,
                                         w.INPUT_MANIFEST_PATH.relative_to(ROOT).as_posix(): w.INPUT_MANIFEST_SHA256})
        record = {}
        with patch.object(w, 'load_checked_source', return_value=SimpleNamespace(prepare=prepare)) as load, \
                patch.dict(sys.modules, {'inference_texture': SimpleNamespace(
                    inputs=SimpleNamespace(require_prepared=require_prepared))}):
            actual = w.prepare_inputs(profile, plan, record)
        self.assertIs(actual, prepared)
        load.assert_called_once_with('brick_qualified_readonly_input', w.INPUT_POLICY_PATH, w.INPUT_POLICY_SHA256)
        prepare.assert_called_once_with(manifest_path=w.INPUT_MANIFEST_PATH,
            attestation_path=w.INPUT_ATTESTATION_PATH, expected_manifest_sha256=w.INPUT_MANIFEST_SHA256,
            expected_attestation_sha256=w.INPUT_ATTESTATION_SHA256)
        require_prepared.assert_called_once_with(prepared, profile)
        self.assertTrue(record['input_route']['generic_array_revalidation'])
        self.assertTrue(record['input_route']['supplied_profile_verified'])
        self.assertTrue(record['input_route']['plan_closure_verified'])
        self.assertIs(record['acceptance_security_results'], security)

    def test_missing_candidate_route_is_rejected_before_loading_the_adapter(self):
        with patch.object(self.worker, 'load_checked_source') as load:
            with self.assertRaisesRegex(ValueError, 'closure'):
                self.worker.prepare_inputs(object(), dict(candidate_identities={}), {})
            load.assert_not_called()

    def test_generic_validation_failure_is_not_promoted_to_qualified_arrays(self):
        w = self.worker
        prepared = dict(profile=object(), metadata={})
        policy = SimpleNamespace(prepare=lambda **_: dict(prepared=prepared, acceptance_security_results={}))
        package = SimpleNamespace(inputs=SimpleNamespace(require_prepared=Mock(side_effect=ValueError('array boundary refusal'))))
        record = {}
        plan = dict(candidate_identities={w.INPUT_POLICY_PATH.relative_to(ROOT).as_posix(): w.INPUT_POLICY_SHA256,
                                         w.INPUT_MANIFEST_PATH.relative_to(ROOT).as_posix(): w.INPUT_MANIFEST_SHA256})
        with patch.object(self.worker, 'load_checked_source', return_value=policy), \
                patch.dict(sys.modules, {'inference_texture': package}), \
                self.assertRaisesRegex(ValueError, 'array boundary refusal'):
            self.worker.prepare_inputs(prepared['profile'], plan, record)
        self.assertNotIn('input_route', record)

    def test_cpu_cannot_skip_the_production_profile_or_closure_branches(self):
        for profile, plan in ((None, None), (object(), None), (None, {})):
            with self.subTest(profile=profile, plan=plan), \
                    patch.object(self.worker, 'load_checked_source') as load, \
                    self.assertRaisesRegex(ValueError, 'Production profile'):
                self.worker.prepare_inputs(profile, plan, {})
            load.assert_not_called()

    def test_foreign_cpu_output_is_refused_without_creating_files(self):
        foreign = HERE/'foreign-CPU-output'
        with patch.object(self.worker, 'load_checked_source') as load:
            with self.assertRaisesRegex(ValueError, 'CPU output'):
                self.worker.cpu_input_acceptance(SimpleNamespace(worker=foreign))
            load.assert_not_called()
        self.assertFalse(foreign.exists())

    def test_pinned_input_sources_and_all_old_closure_members_are_required(self):
        w = self.worker
        self.assertEqual(hashlib.sha256(w.INPUT_POLICY_PATH.read_bytes()).hexdigest(), w.INPUT_POLICY_SHA256)
        self.assertEqual(hashlib.sha256(w.INPUT_MANIFEST_PATH.read_bytes()).hexdigest(), w.INPUT_MANIFEST_SHA256)
        self.assertEqual(hashlib.sha256(w.CPU_GUARD_PATH.read_bytes()).hexdigest(), w.CPU_GUARD_SHA256)
        import json
        required = w.required_candidate_paths()
        self.assertTrue(set(json.loads(w.INPUT_MANIFEST_PATH.read_bytes())).issubset(required))
        self.assertIn(w.INPUT_MANIFEST_PATH.relative_to(ROOT).as_posix(), required)


def load_tests(loader, tests, pattern):
    # Reuse the original real worker control tests against this preserved copy.
    if hashlib.sha256(OLD_TEST.read_bytes()).hexdigest() != '638ff59ea15c529f238f7d1c38ecb537abae07f255307f95ec79cf274df23c48':
        raise ValueError('Historical worker tests changed')
    historical = source_module('qualified_worker_historical_controls', OLD_TEST)
    historical.SOURCE = SOURCE
    tests.addTests(loader.loadTestsFromTestCase(historical.WorkerTests))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)
