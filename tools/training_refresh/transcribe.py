import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import prepare
import transcribe_tokine_added7 as t
t.SONGS=prepare.mod.SONGS;t.PREP=prepare.mod.PREP;t.main()
