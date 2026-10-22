"""Pure counter receipt checks, written before the corrected worker."""
import sys
sys.dont_write_bytecode=True
import importlib.util, unittest
from pathlib import Path
path=Path(__file__).with_name('import_probe.py')
module=None
if path.exists():
    spec=importlib.util.spec_from_file_location('corrected_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def state(value):
    return dict(length=1,values=[value],getter=dict(ok=True,value=value))

class CounterTests(unittest.TestCase):
    def check(self,*args,reject=False):
        self.assertIsNotNone(module,'Corrected counter validator is not implemented')
        if reject:
            with self.assertRaises(ValueError):module.validate_counter(*args)
        else:self.assertTrue(module.validate_counter(*args))
    def test_actual_zero_output_overwrites_nonzero(self):
        self.check(state(-23456),-23456,state(0),0,[])
    def test_nonzero_output_semantics_not_transport_permission(self):
        self.check(state(-23456),-23456,state(2),2,[4,7])
    def test_wrong_sentinel(self):
        self.check(state(0),0,state(0),0,[],reject=True)
    def test_unchanged_holder(self):
        self.check(state(-23456),-23456,state(-23456),-23456,[],reject=True)
    def test_detached_settings_before(self):
        self.check(state(-23456),0,state(0),0,[],reject=True)
    def test_detached_settings_after(self):
        self.check(state(-23456),-23456,state(0),-23456,[],reject=True)
    def test_empty_before(self):
        self.check(dict(length=0,values=[],getter=dict(ok=False)), -23456,state(0),0,[],reject=True)
    def test_multiple_after(self):
        self.check(state(-23456),-23456,dict(length=2,values=[0,0],getter=dict(ok=False)),0,[],reject=True)
    def test_false_getter(self):
        row=state(0);row['getter']['ok']=False
        self.check(state(-23456),-23456,row,0,[],reject=True)
    def test_incoherent_storage(self):
        row=state(0);row['getter']['value']=1
        self.check(state(-23456),-23456,row,0,[],reject=True)
    def test_boolean_counts(self):
        self.check(state(-23456),-23456,state(False),False,[],reject=True)
    def test_float_counts(self):
        self.check(state(-23456),-23456,state(0.0),0.0,[],reject=True)
    def test_negative_output(self):
        self.check(state(-23456),-23456,state(-1),-1,[],reject=True)
    def test_failed_region_disagreement(self):
        self.check(state(-23456),-23456,state(0),0,[4],reject=True)
    def test_duplicate_failed_IDs(self):
        self.check(state(-23456),-23456,state(2),2,[4,4],reject=True)
    def test_bad_failed_IDs(self):
        for value in (-1,True,1.0):
            self.check(state(-23456),-23456,state(1),1,[value],reject=True)

if __name__=='__main__':unittest.main(verbosity=2)
