"""CPU acceptance rejects contradictory sequence, artifact, and provenance evidence."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))

import amd_texture_sampling_contract as c
import amd_texture_sampling_local as local


def candidate():
    modules = {'<root>': 1, **{f'blocks.{i}': 1 for i in range(30)},
               **{f'blocks.{i}.{a}': 1 for i in range(30) for a in ('self_attn', 'cross_attn')}}
    memory = dict(allocated_bytes=1, reserved_bytes=1, peak_allocated_bytes=1, peak_reserved_bytes=1,
                  available_bytes=c.FREE_RESERVE+c.OPAQUE_HEADROOM)
    steps = [dict(step=i + 1, t=t, next_t=c.times()[i + 1], forwards=1, total_forwards=i + 1,
                  all_finite=True, synchronized=True, coords_unchanged=True, elapsed_seconds=1., memory=memory.copy())
             for i, t in enumerate(c.times()[:-1])]
    sha = 'a' * 64
    inputs = {key: {'sha256': sha} for key in ('x', 'coords', 'cond', 'shape_cond')}
    sampling = dict(sequences=1, started=12, completed=12, positive_calls=12, negative_calls=0, concat_calls=12, steps=steps,
                    models=[{**ticket, 'modules': modules, 'frequency': 1, 'cache_reused': ticket['index'] > 1}
                            for ticket in c.tickets()])
    checkpoint = dict(checkpoint_sha256=local.CHECKPOINT_SHA, config_sha256=local.CONFIG_SHA,
                      source_sha256='3959f1f36f17ecd59ba06b91c948ac85303a3aab7d572f6248a846bc50f68d0c',
                      conditioning=dict(verified=True, file_sha256=local.FEATURE_SHA,
                                        output_sha256=sha))
    provenance = dict(harness_commit='b' * 40, checkpoint=checkpoint, input=copy.deepcopy(inputs),
                      parameters={**c.PARAMS, 'guidance_interval': list(c.PARAMS['guidance_interval'])},
                      sigma_min=c.SIGMA_MIN, seed=c.SEED, shape_normalization=c.SHAPE_NORMALIZATION,
                      texture_normalization=c.TEXTURE_NORMALIZATION, sampling=copy.deepcopy(sampling))
    tensors = {key: dict(dtype='I32' if key == 'coords' else 'F32',
                         shape=[3743, 4 if key == 'coords' else 32],
                         tensor_bytes=59888 if key == 'coords' else 479104, sha256=sha)
               for key in ('raw', 'denormalized', 'noise', 'coords')}
    metadata = dict(format='modly-amd-texture-latent-v1', artifact_file='texture-latent.safetensors',
                    artifact_bytes=1500000, artifact_sha256=sha, tensors=tensors, provenance=provenance)
    tunable = dict(environment={key: '0' for key in ('PYTORCH_TUNABLEOP_ENABLED', 'PYTORCH_TUNABLEOP_TUNING',
                                                   'PYTORCH_TUNABLEOP_RECORD_UNTUNED')},
                   flags={key: False for key in ('is_enabled', 'tuning_is_enabled', 'record_untuned_is_enabled')})
    return dict(status='PASS', stage='complete', load_calls=1, forward_calls=12, workspace_clear_calls=1,
                tunable_import={**copy.deepcopy(tunable), 'gpu_initialized': False},
                tunable_before_forward={**copy.deepcopy(tunable), 'gpu_initialized': True},
                workspace_binding={'verified': True}, forward_seconds=20., cpu_load={'elapsed_seconds': 10.},
                output=dict(shape=[3743, 32], dtype='torch.float32', device='cuda:0', all_finite=True,
                            count=3743 * 32, requires_grad=False, is_inference_tensor=True,
                            grad_enabled=False, coords_unchanged=True, sparse_shape=[1, 32], sha256=sha),
                input=inputs, release=dict(allocated_bytes=0, reserved_bytes=0,
                                           model_collected=True, owned_tensors_collected=True),
                calls={key: 12 for key in modules}, sampling=sampling,
                transfers={**{key: 1 for key in ('x', 'coords', 'batch_map', 'seqlen', 'cond', 'shape_cond',
                                                'output', 'output_coords', 'rope_frequency')}, 't': 12, 'frequency': 12},
                artifacts=dict(verified=True, artifact_path='scratch/texture-latent.safetensors',
                               metadata_path='scratch/texture-latent-metadata.json', metadata=metadata))


class ResultTests(unittest.TestCase):
    def test_complete_twelve_positive_evidence(self):
        self.assertEqual(c.classify(candidate(), True, True, True), 'PASS')

    def test_tunable_environment_actual_flags_and_gpu_phase_are_required(self):
        for phase, expected_gpu in [('tunable_import', False), ('tunable_before_forward', True)]:
            missing = candidate()
            missing.pop(phase)
            self.assertEqual(c.classify(missing, True, True, True), 'PARTIAL')
            for flag in ('is_enabled', 'tuning_is_enabled', 'record_untuned_is_enabled'):
                for bad_value in (True, 0, None):
                    row = candidate()
                    row[phase]['flags'][flag] = bad_value
                    self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
            for environment in ('PYTORCH_TUNABLEOP_ENABLED', 'PYTORCH_TUNABLEOP_TUNING',
                                'PYTORCH_TUNABLEOP_RECORD_UNTUNED'):
                for bad_value in ('1', 0, None):
                    row = candidate()
                    row[phase]['environment'][environment] = bad_value
                    self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
            for bad_gpu in (not expected_gpu, int(expected_gpu), None):
                row = candidate()
                row[phase]['gpu_initialized'] = bad_gpu
                self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
            for key in ('flags', 'environment'):
                row = candidate()
                row[phase][key]['foreign'] = False
                self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')

    def test_reject_changed_sequence_timing_and_transfer(self):
        for key, value in [('negative_calls', 1), ('started', 13), ('sequences', True)]:
            row = candidate()
            row['sampling'][key] = value
            self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
        for key, value in [('t', .5), ('elapsed_seconds', float('nan')), ('forwards', 2), ('all_finite', 1)]:
            row = candidate()
            row['sampling']['steps'][0][key] = value
            self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
        row = candidate()
        row['transfers']['neg_cond'] = 1
        self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
        for flags in [(False, True, True), (True, False, True), (True, True, False)]:
            self.assertEqual(c.classify(candidate(), *flags), 'PARTIAL')

    def test_reject_artifact_identity_and_provenance_substitution(self):
        for key, value in [('seed', 20261013), ('sigma_min', 0),
                           ('shape_normalization', c.TEXTURE_NORMALIZATION),
                           ('texture_normalization', c.SHAPE_NORMALIZATION), ('harness_commit', '')]:
            row = candidate()
            row['artifacts']['metadata']['provenance'][key] = value
            self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
        for key in ('checkpoint_sha256', 'config_sha256', 'source_sha256'):
            row = candidate()
            row['artifacts']['metadata']['provenance']['checkpoint'][key] = '0' * 64
            self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')
        for key in ('raw', 'noise', 'coords'):
            row = candidate()
            row['artifacts']['metadata']['tensors'][key]['sha256'] = '0' * 64
            self.assertEqual(c.classify(row, True, True, True), 'PARTIAL')

    def test_exact_memory_cap_and_reserve_boundaries(self):
        row = candidate()
        row['sampling']['steps'][0]['memory']['reserved_bytes'] = c.MEMORY_LIMIT
        row['artifacts']['metadata']['provenance']['sampling'] = copy.deepcopy(row['sampling'])
        self.assertEqual(c.classify(row, True, True, True), 'PASS')
        for key, value in [('reserved_bytes', c.MEMORY_LIMIT+1),
                            ('available_bytes', c.FREE_RESERVE+c.OPAQUE_HEADROOM-1)]:
            bad = copy.deepcopy(row)
            bad['sampling']['steps'][0]['memory'][key] = value
            self.assertEqual(c.classify(bad, True, True, True), 'PARTIAL')


if __name__ == '__main__':
    unittest.main()
