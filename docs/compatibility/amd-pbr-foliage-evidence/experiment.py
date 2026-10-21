"""Exact worker entry selected by the frozen CPU supervisor."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from foliage_worker import main
if __name__=='__main__':main()
