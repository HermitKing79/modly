"""Complete positive-census schema handoff, without original array processing."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest,json,hashlib
import numpy as np
import summary_contract as sc

class SummaryTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).resolve().parents[4]/'artifacts/amd-brick-safe-census-e262ad7/diagnose/result.json'
        self.raw=p.read_bytes();self.parent=json.loads(self.raw);self.s=self.parent['summary']
    def test_complete_authentic_34_field_schema(self):
        sc.validate_summary(self.s);self.assertEqual(len(self.s),34)
        pairs=sc.validated_mapping(sc.MASK_SUMMARY_PAIRS)
        self.assertEqual(pairs['certified_edge_touch_faces'],'certified_edge_face_population')
        self.assertEqual(set(pairs.values())|sc.OTHER_FIELDS,set(self.s))
    def test_actual_complete_mask_count_handoff(self):
        # Bounded original counts, generated boolean vectors only; no census read.
        masks={k:np.ones(self.s[target],np.bool_) for k,target in sc.MASK_SUMMARY_PAIRS}
        sc.verify_mask_counts(self.s,masks)
        for name in masks:
            with self.subTest(name=name):
                bad=dict(masks);bad[name]=np.array([not bool(self.s[dict(sc.MASK_SUMMARY_PAIRS)[name]])],np.bool_)
                with self.assertRaises(ValueError):sc.verify_mask_counts(self.s,bad)
    def test_every_unknown_missing_and_bad_type_field_fails(self):
        for name in self.s:
            for bad_value in (-1,True,1.0,None):
                bad=dict(self.s);bad[name]=bad_value
                with self.subTest(name=name,bad=bad_value),self.assertRaises(ValueError):sc.validate_summary(bad)
            bad=dict(self.s);del bad[name]
            with self.assertRaises(ValueError):sc.validate_summary(bad)
        bad=dict(self.s,unknown=0)
        with self.assertRaises(ValueError):sc.validate_summary(bad)
    def test_all_34_changed_values_fail_sealed_producer_identity(self):
        sha=hashlib.sha256(self.raw).hexdigest();self.assertEqual(sc.sealed_json(self.raw,sha),self.parent)
        for name in self.s:
            bad=json.loads(self.raw);bad['summary'][name]+=1
            with self.subTest(name=name),self.assertRaises(ValueError):sc.sealed_json(json.dumps(bad).encode(),sha)
    def test_duplicate_json_keys_fail_before_dictionary_collapse(self):
        for raw in ('{"summary":{"x":1,"x":2}}','{"summary":{},"summary":{}}'):
            b=raw.encode()
            with self.assertRaisesRegex(ValueError,'Duplicate'):sc.sealed_json(b,hashlib.sha256(b).hexdigest())
    def test_mapping_duplicates_unknown_and_wrong_target_fail(self):
        for pairs in (sc.MASK_SUMMARY_PAIRS+(sc.MASK_SUMMARY_PAIRS[0],),sc.MASK_SUMMARY_PAIRS[:-1],(('unknown','unknown'),)+sc.MASK_SUMMARY_PAIRS[1:],((sc.MASK_SUMMARY_PAIRS[0][0],'certified_edge_face_population'),)+sc.MASK_SUMMARY_PAIRS[1:]):
            with self.assertRaises(ValueError):sc.validated_mapping(pairs)
    def test_incorrect_population_or_work_arithmetic_fails(self):
        for name in ('pre_guard_interior_faces','link_comparisons','batch_edge_limit','pre_link_candidate_edges','referenced_vertices'):
            bad=dict(self.s);bad[name]+=1
            with self.subTest(name=name),self.assertRaises(ValueError):sc.validate_summary(bad)

if __name__=='__main__':unittest.main(verbosity=2)
