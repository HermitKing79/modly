"""Original fixture timing contracts; Popen, sleep and save are replaced.

These synthetic clocks establish source semantics, not timings of the saved run.
No child, Windows Job, WSL or GPU operation is executed.
"""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class Clock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = None

    def monotonic(self): return self.now

    def sleep(self, seconds):
        if self.on_sleep is not None:
            callback, self.on_sleep = self.on_sleep, None
            callback()
        else:
            self.now += seconds


class MemoryFile:
    def __init__(self, name, files): self.name, self.files = name, files
    def exists(self): return self.name in self.files
    def read_text(self): return self.files[self.name]


class MemoryDirectory:
    def __init__(self): self.files = {}
    def __truediv__(self, name): return MemoryFile(name, self.files)


class FixtureTimingContracts(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'docs/compatibility/amd-job-stage-orchestrator-evidence/cpu_stage_fixture.py'
        spec = importlib.util.spec_from_file_location('original_fixture_contract', path)
        self.fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.fixture)
        self.base, self.clock, self.saved = MemoryDirectory(), Clock(), []
        self.child_waits = []
        self.spawn_delay = 0

    def publish(self, *names):
        for name in names:
            self.base.files[name] = json.dumps({'pid': 23, 'grandchild': 24})

    def run_parent(self):
        owner = self
        class Child:
            def wait(self, timeout): owner.child_waits.append(timeout); return 0
        def spawn(*args, **kwargs):
            self.clock.now += self.spawn_delay
            return Child()
        with patch.object(self.fixture.Path, 'cwd', return_value=self.base), \
             patch.dict(os.environ, {'MODLY_COMPLETION_CONTEXT_JSON': '{"nonce":"synthetic"}', 'MODLY_STAGE_ID': 'contract'}), \
             patch.object(self.fixture.subprocess, 'Popen', side_effect=spawn), \
             patch.object(self.fixture.time, 'monotonic', self.clock.monotonic), \
             patch.object(self.fixture.time, 'sleep', self.clock.sleep), \
             patch.object(self.fixture, 'save', side_effect=lambda path, row: self.saved.append(path.name)):
            self.fixture.main('success')

    def test_absent_markers_timeout_at_inclusive_five_second_check(self):
        self.clock.on_sleep = lambda: setattr(self.clock, 'now', 5.0)
        with self.assertRaisesRegex(TimeoutError, 'Fixture descendants did not start'): self.run_parent()
        self.assertEqual(self.clock.now, 5.0)
        self.assertNotIn('ready.json', self.saved)
        self.assertEqual(self.child_waits, [])

    def test_child_marker_alone_does_not_prove_both_ready(self):
        def delay(): self.publish('child-ready.json'); self.clock.now = 6.0
        self.clock.on_sleep = delay
        with self.assertRaisesRegex(TimeoutError, 'Fixture descendants did not start'): self.run_parent()
        self.assertTrue((self.base / 'child-ready.json').exists())
        self.assertNotIn('ready.json', self.saved)

    def test_both_markers_visible_after_sleep_are_accepted_even_past_deadline(self):
        def delay():
            self.clock.now = 6.0
            self.publish('child-ready.json', 'grandchild-ready.json')
        self.clock.on_sleep = delay
        self.run_parent()
        self.assertGreater(self.clock.now, 5)
        self.assertIn('ready.json', self.saved)
        self.assertIn('stop.json', self.saved)

    def test_popen_duration_precedes_and_is_excluded_from_five_second_budget(self):
        self.spawn_delay = 8
        self.clock.on_sleep = lambda: setattr(self.clock, 'now', 13.0)
        with self.assertRaisesRegex(TimeoutError, 'Fixture descendants did not start'): self.run_parent()
        self.assertEqual(self.clock.now, 13.0)


if __name__ == '__main__': unittest.main()
