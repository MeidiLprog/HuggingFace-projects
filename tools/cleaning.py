import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from smolagents import Tool
import numpy as np

class CleaningTool(Tool):
    name = "clean_data"
    description = (
        "Clean a pandas DataFrame: remove duplicates, fill missing values, "
        "encode categorical columns, and detect low-variance features."
    )
    inputs = {
        "data": {
            "type": "object",
            "description": "Pandas DataFrame or dict-like object to clean"
        }
    }
    output_type = "object" 


    def forward(self,data):

        df = data.copy()

        print("First step: We are to take a look at our data\n")
        print(f"info {df.info()}\n")
        print(f"info {df.describe(include='all')}\n")

        miss_info = df.isnull().sum()
        print("Missing values per column \n")
        print(f"{miss_info}\n")

        print("\n Let us check for duplicates \n")
        duplicate = df.duplicated().sum()
        if duplicate != 0:
            print(f"Duplicated detected {duplicate}\n")
        else:
            print("No duplicates how lucky !\n")

        
        
        for col in df.columns:
            if df[col].dtype == "object": #categorial variable
                if df[col].isnull().any(): #any null values
                    print("Imputation for the qualitative variable here")
                    df[col].fillna(df[col].mode()[0], inplace=True)
                print("LabelEncoding...\n")
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            else:
                if df[col].isnull().any():
                    print("Imputation using Median\n")
                    df[col].fillna(df[col].median(),inplace=True)

        num_cols = df.select_dtypes(exclude=["object","datetime64[ns]"]).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        low_var = [col for col in num_cols if df[col].nunique() <= 1]
        if low_var:
            print(f"columns with a low variance {low_var} \n")
            df.drop(columns=low_var,inplace=True)
        
        cat_col = df.select_dtypes(include="object").columns
        if len(cat_col) > 0:
            encoder = OneHotEncoder(sparse=False,drop='if_binary')
            encode = encoder.fit_transform(df[cat_col])
            encoded_df = pd.DataFrame(encode,columns=encoder.get_feature_names_out(cat_col))
            df = pd.concat([df.drop(columns=cat_col).reset_index(drop=True), encoded_df], axis=1)
        print("Cleaning done\n")

        return df


