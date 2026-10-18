"""Focused host qualification; no WSL, model, installation or GPU execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location('qualified_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), vars(module))
    return module


host = load('host')
cpu = load('cpu_acceptance')
worker = load('worker')


class HostTests(unittest.TestCase):
    def test_original_resource_and_deadline_controls_stay_exact(self):
        self.assertEqual(host.PRELAUNCH_RAM, 37 * 1024**3 // 2)
        self.assertEqual(host.LIVE_RAM, 12 * 1024**3)
        self.assertEqual(host.IO_SECONDS, 30)
        self.assertEqual(host.DEADLINES, dict(native=60, cpu=120, gpu=240, host=480))
        self.assertEqual(host.SCRATCH, '/home/modly/brick-pbr-qualified-a064378')
        self.assertEqual(host.DESTINATION.name, 'amd-brick-pbr-qualified-a064378')

    def test_full_plan_constructs_before_publication_and_authenticates(self):
        # Exercise the actual complete plan builder, Windows serializer and guest
        # schema validator. Only filesystem/source admission is substituted.
        profile = SimpleNamespace(sha256='c'*64, run_contract_sha256='d'*64)
        proof = dict(mode='production', profile_sha256=profile.sha256,
                     run_contract_sha256=profile.run_contract_sha256)
        proof_blob = json.dumps(proof).encode()
        proof_sha = hashlib.sha256(proof_blob).hexdigest()
        original_digest = host.digest
        def digest(path):
            if Path(path) == host.PROBE / 'ownership.json':
                return proof_sha
            if Path(path) == HERE / 'candidate-identities.json':
                return 'e'*64
            return original_digest(path)
        with patch.object(host, 'HARNESS_COMMIT', 'a'*40), \
                patch.object(host, 'candidate_files', return_value={host.WORKER.relative_to(host.ROOT).as_posix(): 'f'*64}), \
                patch.object(host, 'revalidate_qualified_inputs', return_value={'passed': True}), \
                patch.object(host, 'plan_worker', return_value=SimpleNamespace(verify_manifest=lambda *_: None)), \
                patch.object(host.receipts, '_leaves'), \
                patch.object(host, 'selected_profile', return_value=profile), \
                patch.object(host.inputs, 'prepare', return_value={}), \
                patch.object(host, 'read', return_value=proof), \
                patch.object(host, 'digest', side_effect=digest), \
                patch.object(host.receipts, '_publish') as publish:
            marker, plan, evidence, marker_sha, plan_sha = host.prepare_attempt(HERE/'run')
        publish.assert_not_called()
        self.assertEqual(len(plan), 13)
        self.assertEqual(marker_sha, hashlib.sha256(host.serialized_publication(marker)).hexdigest())
        self.assertEqual(plan_sha, hashlib.sha256(host.serialized_publication(plan)).hexdigest())
        self.assertEqual(evidence['plan_sha256'], plan_sha)
        self.assertEqual(plan['geometry_contract_file_sha256'], host.GEOMETRY_CONTRACT_SHA256)
        guest_ev = Path(host.to_guest(HERE))
        guest_probe = Path(host.to_guest(host.PROBE))
        blobs = {guest_probe/'execution-plan.json': host.serialized_publication(plan),
                 guest_probe/'neural-attempt.marker': host.serialized_publication(marker),
                 guest_probe/'ownership.json': proof_blob,
                 Path(plan['geometry_contract_path']): host.GEOMETRY_CONTRACT.read_bytes()}
        with patch.object(worker, 'ROOT', Path(host.to_guest(host.ROOT))), \
                patch.object(worker, 'EV', guest_ev), \
                patch.object(worker, 'FIELD', Path(host.to_guest(host.FIELD))), \
                patch.object(worker, 'PROBE', guest_probe), \
                patch.object(worker, 'read_plain', side_effect=lambda path, _: blobs[Path(path)]), \
                patch.object(worker, 'verify_manifest', return_value={'qualified': 'f'*64}):
            authenticated = worker.authenticate_plan(guest_probe/'execution-plan.json', plan_sha, guest_ev/'run/guest')
        self.assertEqual(authenticated['execution_plan_sha256'], plan_sha)
        self.assertEqual(authenticated['candidate_identities'], {'qualified': 'f'*64})

    def test_changed_preserved_source_closure_is_rejected(self):
        preserved = {'old/source.py': 'a'*64}
        with patch.object(host, 'read', side_effect=[{}, preserved]), \
                patch.object(host, '_read_plain', return_value=b'preserved'), \
                patch.object(host, 'HISTORICAL_MANIFEST_SHA256', hashlib.sha256(b'preserved').hexdigest()):
            with self.assertRaisesRegex(ValueError, 'Preserved source closure incomplete'):
                host.candidate_files()

    def test_fresh_attestation_uses_exact_checked_source_and_proof(self):
        checked = []
        blob = b'def verify(path, expected):\n return {"passed": True, "files": 5, "path": str(path), "sha256": expected}\n'
        def read(path, maximum):
            checked.append(Path(path))
            return blob
        sha = hashlib.sha256(blob).hexdigest()
        with patch.object(host, '_read_plain', side_effect=read), \
                patch.object(host, 'INPUT_POLICY_SHA256', sha), \
                patch.object(host, 'INPUT_MANIFEST_SHA256', sha), \
                patch.object(host, 'WINDOWS_ATTEST_SHA256', sha):
            result = host.revalidate_qualified_inputs()
        self.assertEqual(result['path'], str(host.INPUT_ATTESTATION))
        self.assertEqual(result['sha256'], host.INPUT_ATTESTATION_SHA256)
        self.assertEqual(set(checked), {host.INPUT_POLICY, host.INPUT_MANIFEST, host.WINDOWS_ATTEST})

    def test_bad_attestation_source_cannot_reach_verifier(self):
        with patch.object(host, '_read_plain', return_value=b'wrong'):
            with self.assertRaisesRegex(ValueError, 'Qualified input source identity differs'):
                host.revalidate_qualified_inputs()

    def test_cpu_command_targets_actual_production_worker_under_modly(self):
        command = cpu.build_worker_command({'worker.py': 'a'*64, 'cpu_acceptance.py': 'b'*64, 'manifest': 'e'*64})
        self.assertEqual(command[:7], ['wsl.exe', '-d', cpu.DISTRO, '-u', 'modly', '--exec', '/usr/bin/env'])
        self.assertIn('--cpu-input-acceptance', command)
        self.assertIn(cpu.guest_path(worker.SCRIPT), command)
        self.assertIn(cpu.guest_path(cpu.OUTPUT/'guest'), command)
        self.assertIn('90s', command)
        self.assertIn('-I', command)
        self.assertIn('-B', command)
        self.assertNotIn('--plan', command)
        self.assertNotIn('--manifest', command)
        self.assertTrue(all(key+'=0' in command for key in cpu.TUNABLE_ENV))
        self.assertEqual(command[command.index('--expected-host-sha256')+1], 'b'*64)
        self.assertEqual(command[command.index('--expected-manifest-sha256')+1], 'e'*64)

    def valid_cpu_record(self):
        route = dict(production_prepare_function='prepare_inputs', generic_array_revalidation=True,
            policy_path=cpu.guest_path(cpu.INPUT_POLICY), policy_sha256=cpu.INPUT_POLICY_SHA256,
            manifest_path=cpu.guest_path(cpu.INPUT_MANIFEST), manifest_sha256=cpu.INPUT_MANIFEST_SHA256,
            attestation_path=cpu.guest_path(cpu.PRIVATE_ATTESTATION), attestation_sha256=cpu.PRIVATE_ATTESTATION_SHA256,
            historical_readers_modified=False, writer_capability=False,
            supplied_profile_verified=True, plan_closure_verified=True)
        return dict(classification='UNKNOWN', worker_returncode=0, observer_healthy=True,
            windows_job_cleanup=dict(owned_tree_exit_verified=True, forced_job_cleanup=False),
            windows_job_grace=dict(empty=True), distro_termination=dict(returncode=0), elapsed_seconds=1,
            worker=dict(classification='CPU_PREPARED', gate_authorized=True, CPU_only=True,
                forbidden_import_attempts=[], forbidden_modules=[], input_route=route,
                source_manifest_sha256='e'*64, candidate_identities={'real/source.py': 'f'*64},
                authenticated_profile_sha256=cpu.PROFILE_SHA256,
                runtime=dict(uid=1000, euid=1000, gid=1000, egid=1000),
                prepared_metadata=dict(cpu_only=True, profile_sha256=cpu.PROFILE_SHA256,
                    run_contract_sha256=cpu.RUN_CONTRACT_SHA256), prepared_arrays_released=True,
                prepared_array_objects_observed=4,
                acceptance_security_results=dict(all_passed=True, tests=[dict(passed=True)])))

    def test_cpu_verdict_demands_actual_preparation_and_clean_containment(self):
        raw = self.valid_cpu_record()
        self.assertEqual(cpu.cpu_verdict(1, raw, 'e'*64)['classification'], 'PASS')
        mutations = (
            lambda row: row['windows_job_cleanup'].update(forced_job_cleanup=True),
            lambda row: row['worker']['runtime'].update(uid=0),
            lambda row: row['worker']['input_route'].update(writer_capability=True),
            lambda row: row['worker']['input_route'].update(production_prepare_function='fake'),
            lambda row: row['worker']['input_route'].update(supplied_profile_verified=False),
            lambda row: row['worker']['input_route'].update(plan_closure_verified=False),
            lambda row: row['worker'].update(source_manifest_sha256='a'*64),
            lambda row: row['worker'].update(candidate_identities={}),
            lambda row: row['worker'].update(authenticated_profile_sha256='a'*64),
            lambda row: row['worker']['prepared_metadata'].update(profile_sha256='a'*64),
            lambda row: row['worker'].update(prepared_arrays_released=False),
            lambda row: row['worker']['acceptance_security_results'].update(tests=[]),
            lambda row: row['worker'].update(forbidden_modules=['torch']),
        )
        for change in mutations:
            rejected = copy.deepcopy(raw)
            change(rejected)
            self.assertEqual(cpu.cpu_verdict(1, rejected, 'e'*64)['classification'], 'UNKNOWN')

    def test_cpu_checked_source_rejects_modified_digest(self):
        with tempfile.TemporaryDirectory(prefix='brick-qualified-host-', dir=host.ROOT.parent/'artifacts') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to((host.ROOT.parent/'artifacts').resolve()))
            path = Path(directory)/'source.py'
            path.write_bytes(b'fixed source')
            self.assertEqual(cpu.checked_source(path, hashlib.sha256(b'fixed source').hexdigest()), b'fixed source')
            with self.assertRaisesRegex(ValueError, 'Pinned source identity differs'):
                cpu.checked_source(path, '0'*64)

    def test_historical_input_sources_are_exact_qualified_bytes(self):
        for path, sha in ((host.INPUT_POLICY, host.INPUT_POLICY_SHA256),
                          (host.WINDOWS_ATTEST, host.WINDOWS_ATTEST_SHA256),
                          (host.INPUT_MANIFEST, host.INPUT_MANIFEST_SHA256),
                          (host.HISTORICAL_MANIFEST, host.HISTORICAL_MANIFEST_SHA256)):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)

    def test_cpu_preflight_can_authenticate_the_complete_preserved_closure(self):
        preserved = json.loads(host.HISTORICAL_MANIFEST.read_bytes())
        preserved.update(json.loads(host.INPUT_MANIFEST.read_bytes()))
        for name, sha in preserved.items():
            cpu.checked_source(host.ROOT/name, sha, maximum=4*1024**2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
