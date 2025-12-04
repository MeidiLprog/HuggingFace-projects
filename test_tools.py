import pandas as pd
from tools.cleaning import CleaningTool
from tools.inspect import InspectTool
from tools.train import TrainTool



df = pd.DataFrame({
    "age" : [20,300,0,350,None,22,1560],
    "Ville": ["Paris","Nice",None,"Alger",None,None,"Tlemcen"],
    "Revenu": [None,None,500,None,600,522,1965],
    "date": pd.date_range(start="2025-12-03 00:00:00", periods=7, freq="8H"),
    "constant" : [1,1,1,1,1,1,1]

})

print("Original Dataset\n")
print(df)
tool = CleaningTool()
df_cleaned = tool.forward(df)
ins = InspectTool()
done_inspection = ins.forward(df)

print("Dataset fully cleaned ! \n")
print(df_cleaned)
