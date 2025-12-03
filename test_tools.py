import pandas as pd
from tools.cleaning import CleaningTool



df = pd.DataFrame({
    "age" : [20,25,30,35,None,22],
    "Ville": ["Paris","Nice",None,"Alger",None,None],
    "Revenu": [None,None,500,None,600,522],
    "constant" : [1,1,1,1,1,1]

})

print("Original Dataset\n")
print(df)
tool = CleaningTool()
df_cleaned = tool.forward(df)
print("Dataset fully cleaned ! \n")
print(df_cleaned)
