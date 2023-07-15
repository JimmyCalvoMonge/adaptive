# base_path
from inspect import currentframe, getframeinfo
from pathlib import Path
import os
filename = getframeinfo(currentframe()).filename
file_path_parent = Path(filename).resolve().parent # ./code/disaggregated/source_code
base_path = os.path.dirname(os.path.dirname(file_path_parent)) # ./code