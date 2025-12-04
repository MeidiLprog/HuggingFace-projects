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
    model = LiteLLMModel(model_id="ollama/qwen2:1.5b")
    agent = CodeAgent(
        model=model,
        tools=[InspectTool(),CleaningTool(),TrainTool()],
        name="autodata_agent",
        description="HuggingFace agent created to carry out datascience/Ml tasks"
    )

    return agent