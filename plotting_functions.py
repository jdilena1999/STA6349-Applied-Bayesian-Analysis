import pandas as pd
import numpy as np

def get_line_coordinates(data:pd.DataFrame,x_col:str,intercept:float,slope:float)->pd.DataFrame:
    x_start,x_end = (min(data[x_col].to_numpy()),max(data[x_col].to_numpy()))
    y = [x*slope+intercept for x in np.linspace(x_start,x_end+1)]
    x = np.linspace(x_start,x_end+1)
    return pd.DataFrame({
        "x":x,
        "y":y
    })