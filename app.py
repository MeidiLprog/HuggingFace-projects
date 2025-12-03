############################
## Lefki Meidi            ##
## First HuggingFace Agent##
############################

import pandas as pd
from agent_logic import create_agent

url : str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)