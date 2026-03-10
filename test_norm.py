import polars as pl
import sys
sys.path.append(r"e:\MAX\PYTHON\NEURAL-BOTS\neirobot-lit")
from python_lab.src.normalization import Normalizer
df = pl.DataFrame({"vib": [1.0, 2.0], "past_ret_10": [0.1, 0.2]})
norm = Normalizer()
try:
    norm.fit(df)
    print("params:", norm.params)
    norm.transform(df)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
