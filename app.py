############################
## Lefki Meidi            ##
## First HuggingFace Agent##
############################

import pandas as pd
from agent_logic import create_agent

url : str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

agent = create_agent()
print("Agent ready !")

agent.run(
    "Inspect, clean and train model to predict survival on the dataset",
    data=df,
    target_name="Survived"

)