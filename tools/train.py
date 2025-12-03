import numpy as np
from smolagents import CodeAgent,LiteLLMModel
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from sklearn.model_selection import GridSearchCV,train_test_split

class TrainTool(Tool):
    name = "train_model"
    description = ("Train a ML model automatically, regressor or classifier")

    inputs = {
 
        "data" : {"type" : "object", "description" : "cleaned pandas dataframes"},
        "target" : {"type" : "string", "description" : "Target column name (optional here)"}
        }
    
    output_type = "string"


    def forward(self,data,name_com=None):
        pass

