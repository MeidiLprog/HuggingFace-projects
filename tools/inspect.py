from smolagents import Tool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class InspectTool(Tool):

    name="visu_data"
    description=("Complete EDA\n")
    inputs = {"data":
             {
                 "type":"object",
                 "description": "Pandas dataframes used to analyse our data"
             }}
    output_type = "string"

    def forward(self, data : pd.DataFrame):

       
        if not isinstance(data,dict):
            data = pd.DataFrame(data)
        elif not isinstance(data,pd.DataFrame):
            raise TypeError("Data must be a pandas dataframe")
        df = data.copy()

        print("Dataset inspection \n")
        print(f"Shape{df.shape}\n")
        print(f"Type of data{df.dtypes}\n")
        print(f"Missing values{df.isnull().sum()}")

        print("Descriptive statistics ! \n")
        try:
            print(f"Describe: {df.describe(include='all', datetime_is_numeric=True)}")
        except TypeError:
            print(f"Describe: {df.describe(include='all')}")
        
        num_cols = df.select_dtypes(exclude=["object","datetime64[ns]"]).columns
        num_cols = [c for c in num_cols if df[c].nunique() > 1]
        cat_cols = df.select_dtypes(exclude=["number"]).columns

        print("Generated visu \n")

        print("Distribution \n")

        if len(num_cols) > 0:
            df[num_cols].hist(bins=20,figsize=(6,12),color="#69b3a2", edgecolor="Black")
            plt.suptitle("Numerics distribution",fontsize=14)
            plt.tight_layout()
            plt.show()

        print("Boxplot \n")

        for col in num_cols:
            plt.figure(figsize=(6,4))
            sns.boxplot(y=df[col],color="#69b3a2")
            plt.ylabel(col)
            plt.xticks(rotation=45)
            plt.title(f"boxplot of {col}\n",fontsize=12)
            plt.tight_layout()
            plt.show()


        
        print("Heatmap if columns are in range of 1-30 at most\n")
        if len(num_cols) > 1 and len(num_cols) <= 30:
            corr = df[num_cols].corr()
            plt.figure(figsize=(6,12))
            sns.heatmap(corr,annot=True,cmap="coolwarm",center=0)
            plt.title("Heatmap")
            plt.tight_layout()
            plt.show()
        print("Visualisation carried out !\n")
        return "Done"


