import numpy as np
from smolagents import CodeAgent,LiteLLMModel
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,mean_squared_error,r2_score
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
        
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a dictionary or a pandas DataFrame")

        df = data.copy()

        if not isinstance(target_name,str):
            raise TypeError("The name of the target must be of type string \n")
        if not target_name:
            target_name = df.columns[-1]
            print(f"No target provided, using last column: {target_name}")




        X = df.drop(columns=[target_name])
        y = df[target_name]

        X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)


        print("Let us figure out whether it's a classification or regression problem \n")

        problem = "Classification" if y.nunique() <= 15 and y.dtype in [object, "int64", "category"] else "Regression"


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
        print("GridSearch training\n")

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        if problem == "Classification":
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
            print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
            print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
        else:
            print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
            print(f"R²: {r2_score(y_test, y_pred):.4f}")

            
        return "training done"        


