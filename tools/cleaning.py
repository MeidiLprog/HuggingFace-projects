import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from smolagents import Tool
import numpy as np

class CleaningTool(Tool):
    name = "Thoroughly clean your data"
    description = ("Analyse data and/or remove duplicates, nans,"
                   "Variables types, encoding, columns suppressions"
                   )
    inputs = {"data" : "Pandas DataFrame"}
    output_type = "Pandas DataFrame"


    def forward(self,data):

        df = data.copy()

        print("First step: We are to take a look at our data\n")
        print(f"info {df.info()}\n")
        print(f"info {df.describe(include='all',datetime_is_numeric=True)}\n")

        miss_info = df.isnull().sum()
        print("Missing values per column \n")
        print(f"{miss_info}\n")

        print("\n Let us check for duplicates \n")
        duplicate = df.duplicated().sum()
        if duplicate != 0:
            print(f"Duplicated detected {duplicate}\n")
        else:
            print("No duplicates how lucky !\n")

        num_cols = df.select_dtypes(exclude=["object"]).columns
        
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

        low_var = [col for col in num_cols if df[col].nunique <= 1]
        if low_var:
            print(f"columns with a low variance {low_var} \n")
        print("Cleaning done\n")

        return df


