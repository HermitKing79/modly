"""Mock-only witnesses: never take a live reading or launch WSL/GPU."""
import importlib.util
from pathlib import Path
import unittest

ENTRY=Path(__file__).with_name('amd_pixal3d_hip_admission.py')
spec=importlib.util.spec_from_file_location('_pixal_admission_test',ENTRY) if ENTRY.exists() else None
subject=None
if spec:
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)

class Exists(unittest.TestCase):
    def test_admission_only_recorder_exists(self):self.assertTrue(ENTRY.is_file(),'Admission-only recorder is missing')

@unittest.skipUnless(subject,'Recorder not implemented yet')
class Gates(unittest.TestCase):
    def readings(self,values):
        values=iter(values);calls=[]
        def read():
            calls.append(1);return dict(available_bytes=next(values))
        return read,calls
    def test_first_low_value_stops_without_second_read_or_sleep(self):
        read,calls=self.readings([22*1024**3-1,24*1024**3]);sleeps=[]
        r=subject.collect(read,sleeps.append)
        self.assertEqual(r['classification'],'NO_GO_WINDOWS_ADMISSION')
        self.assertEqual(len(calls),1);self.assertEqual(sleeps,[])
        self.assertEqual(len(r['readings']),1);self.assertFalse(r['WSL_launched'])
    def test_exact_22_passes_three_readings_with_two_four_second_waits(self):
        read,calls=self.readings([22*1024**3]*3);sleeps=[]
        r=subject.collect(read,sleeps.append)
        self.assertEqual(r['classification'],'PASS_WINDOWS_ADMISSION')
        self.assertEqual(len(calls),3);self.assertEqual(sleeps,[4,4]);self.assertFalse(r['preferred_is_mandatory'])
    def test_later_failure_does_not_take_third_read(self):
        read,calls=self.readings([24*1024**3,21*1024**3,24*1024**3]);sleeps=[]
        r=subject.collect(read,sleeps.append)
        self.assertEqual(r['classification'],'NO_GO_WINDOWS_ADMISSION')
        self.assertEqual(len(calls),2);self.assertEqual(sleeps,[4])
    def test_measurement_error_cannot_admit(self):
        def read():raise OSError('Synthetic reader failure')
        r=subject.collect(read,lambda _:self.fail('Must not wait'))
        self.assertEqual(r['classification'],'NO_GO_WINDOWS_ADMISSION');self.assertEqual(r['readings'],[])
        self.assertFalse(r['native_operations_executed'])

if __name__=='__main__':unittest.main()
