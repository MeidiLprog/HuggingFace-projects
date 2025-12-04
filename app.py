############################
## Lefki Meidi            ##
## First HuggingFace Agent##
############################

import pandas as pd
from agent_logic import create_agent

url : str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Dataset successfully loaded ! {df.shape}")
agent = create_agent()
print("Agent ready !")

message = f"""
You are a data science assistant.

Here is a small preview of the dataset (first 10 rows):
{df.head(10).to_string()}

Your tasks:
1. Inspect and describe the dataset (shape, types, missing values, distributions).
2. Clean it (impute missing data, encode categorical features, normalize if needed).
3. Train a predictive model to classify the 'Survived' column.
4. Display training results and performance metrics.
"""


agent.run(message)