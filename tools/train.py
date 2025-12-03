import numpy as np
from smolagents import CodeAgent,LiteLLMModel
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from sklearn.model_selection import GridSearchCV,train_test_split
import pandas as pd

class TrainTool(Tool):
    name = "train_model"
    description = ("Train a ML model automatically, regressor or classifier")

    inputs = {
 
        "data" : {"type" : "object", "description" : "cleaned pandas dataframes"},
        "target" : {"type" : "string", "description" : "Target column name (optional here)"}
        }
    
    output_type = "string"


    def forward(self,data,target_name=None):
        
        if isinstance(data,dict):
            data = pd.DataFrame(data)
        elif not isinstance(data,dict):
            raise TypeError("Data must be a dictionnary or a dataframe")

        df = data.copy()

        if not isinstance(target_name,str):
            raise TypeError("The name of the target must be of type string \n")
        if (len(target_name) == 0) or (target_name == None):
            target_name = df[-1]



        X = df.drop(columns=[target_name],inplace=True)
        y = df[target_name]

        X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)


        print("Let us figure out whether it's a classification or regression problem \n")

        problem = "Classification" if y.nunique() <= 15 and y.dtype in [object,"int64","category"] else "regression"

        if problem == "Classification":
            model = RandomForestClassifier(random_state=42)
            param_grid = {"n_estimators" : [10,35,40,100], 
                          "max_depth" : [None,10,20,25,26,30,32,36]}

        else:
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                "n_estimators" : [10,35,40,100],
                "max_depth" : [None,10,20,25,26,30,32,36]
            }

        grid = GridSearchCV(model,param_grid,cv=3,n_jobs=-1)
        grid.fit(X_train,y_train)
        

        return "training done"        


