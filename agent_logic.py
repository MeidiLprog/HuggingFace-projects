############################
## Lefki Meidi            ##
## First HuggingFace Agent##
############################

import smolagents
from smolagents import CodeAgent, LiteLLMModel

from tools.inspect import InspectTool
from tools.cleaning import CleaningTool
from tools.train import TrainTool    

def create_agent():
    model = LiteLLMModel(model_id="huggingface/mistralai/Mistral-7B-Instruct-v0.2")
    agent = CodeAgent(
        model=model,
        tools=[InspectTool(),CleaningTool(),TrainTool()],
        name="autodata_agent",
        description="HuggingFace agent created to carry out datascience/Ml tasks"
    )

    return agent