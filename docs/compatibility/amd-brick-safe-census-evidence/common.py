"""Original-only arrays and preserved historical input closure."""
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2]
P=ROOT.parent/'artifacts/amd-brick-safe-census-e262ad7'
OLD=ROOT/'docs/compatibility/amd-brick-junction-protected-evidence'

def pin(name,path,sha):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Changed source: '+str(path))
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

previous=pin('census_previous_common',OLD/'common.py','5dc87d3d6a87036a984de5d9792630350178801276a150d972b24da2a921e34c')
plain,digest,require,save=previous.plain,previous.digest,previous.require,previous.save
original=previous.original

def authenticate():
    watched=previous.authenticate()+[OLD/'common.py']
    for path,sha in {
        OLD/'outcome.json':'05849a3004b22cd4e947b7fa447465833770ddc67ae1f5872fcd84964b5c85e5',
        OLD/'independent-outcome.json':'12c93ce480efd936d3ea96f36209fa45fdcf10b85cee1969877254384e9e0611',
        ROOT/'docs/compatibility/amd-brick-junction-protected-proof.md':'e85fb6e18e3d4e5de346b9cb657e857fa728f49d92cceda8bb6a4c4f536f5c99',
        OLD/'patch_contract.py':'336e49bd7836f4a66c02ade831d6b37b66f290cb42e3aa30c72e7a009599d60c',
    }.items():require(digest(plain(path))==sha,'Previous contract identity');watched.append(path)
    return list(dict.fromkeys(watched))

def begin():previous.P=P;previous.begin()
def load(np):return previous.load(np)
