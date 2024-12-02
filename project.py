from metaflow import FlowSpec, step, card, pypi_base, Parameter, retry, catch, current
from metaflow.cards import Image

import mlflow
import mlflow.sklearn

import os
import copy


# ****** Install all mentioned Libraries ******
@pypi_base(
    packages={
        "scikit-learn": "1.5.2",
        "pandas": "2.2.2",
        "numpy": "2.1.3",
        "scipy": "1.14.1",
        "matplotlib": "3.9.2",
        "seaborn": "0.13.2",
        "mlflow": "2.18.0"
    }
)


# ********* Machine Learning Project with MLOps ***********
class DiseaseClassifier(FlowSpec):
    
    # *** Metaflow Parameters (input from user) ***
    dataset_path = Parameter("dataset_path", help="Getting CSV dataset.", default=os.getcwd() + "/Dataset/heart_disease_uci.csv")
    
    @step
    def start(self):
        """Start Step is mandatory, for Metaflow unified Workflow"""
        self.next(self.loading_data)
    
    
    # *** Metaflow Step with Card Visualization ***
    @card
    @step
    def loading_data(self):
        """ Loading Data from CSV and converting into Pandas DataFrame."""
        import pandas as pd
        
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset_shape = self.dataset.shape
        
        
        self.duplicate_rows = self.dataset.duplicated().sum()
        self.num_null_columns = self.dataset.isnull().sum()
        
        # Branching: Parallel Execution of steps.
        self.next(self.dataset_processing)
        
    
    
    @card
    @catch
    @step
    def dataset_processing(self):
        """ Removing Columns with larger Null values and filling rest columns null value.s"""
        # Removing Columns with larger Null Values
        self.dataset.drop(["ca", "thal"], axis=1, inplace=True)
        
        # Filling Null Values with mean value.
        self.dataset["oldpeak"].fillna(self.dataset["oldpeak"].mean(),  inplace = True)
        self.dataset["thalch"].fillna(self.dataset["thalch"].mean(),  inplace = True)
        self.dataset["trestbps"].fillna(self.dataset["trestbps"].mean(),  inplace = True)
        self.dataset["chol"].fillna(self.dataset["chol"].mean(),  inplace = True)
        self.dataset["fbs"].fillna(False,  inplace = True)
        self.dataset["restecg"].fillna("normal",  inplace = True)
        self.dataset["exang"].fillna(False,  inplace = True)
        self.dataset["slope"].fillna("flat",  inplace = True)
        
        self.dataset = self.dataset
        self.dataset_shape = self.dataset.shape
        self.duplicate_rows = self.dataset.duplicated().sum()
        self.num_null_columns = self.dataset.isnull().sum()
        
        self.next(self.feature_engineering)
        
     
        
    @card
    @catch
    @step
    def feature_engineering(self):
        """ Feature engineering using one-hot encoding."""
        import pandas as pd
        
        # Boolean into Int
        self.dataset["fbs"] = self.dataset["fbs"].astype("int")
        self.dataset["exang"] = self.dataset["exang"].astype("int")

    
        
        # Object features into Numerical Encoded features
        temp_df = self.dataset
        
        labels=['asymptomatic', 'non-anginal', 'atypical angina', 'typical angina']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["cp"] = self.dataset["cp"].map(mapping)
        
        labels=['Male', 'Female']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["sex"] = self.dataset["sex"].map(mapping)
        
        labels=['downsloping', 'flat', 'upsloping']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["slope"] = self.dataset["slope"].map(mapping)
        
        labels = ['lv hypertrophy', 'normal', 'st-t abnormality']
        mapping = {label: i for i, label in enumerate(labels)}
        temp_df["restecg Encoded"] = self.dataset["restecg"].map(mapping)

        
        # One-hot Encoding on Categorical Data
        temp_df = pd.get_dummies(temp_df, columns=["sex", "exang", 'fbs', 'cp', 'slope'])

        self.preparedData = temp_df
        self.next(self.model_training_preprocess)
        
    
    
    
    @card
    @step
    def model_training_preprocess(self):
        
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        
        matrix = show_matrix(self.preparedData)
        current.card.append(Image.from_matplotlib(matrix))
        plt.close(matrix)
        
        # Training and Testing Data
        self.X = self.preparedData.drop(['id', 'dataset', 'restecg' ,'restecg Encoded'], axis = 1)
        self.Y = self.preparedData['restecg']
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=10)
        
        # self.models = ["logisticRegression", "decisionTree", "RandomForest"]
        
        
        self.next(self.logistic_regression, self.decision_tree, self.random_forest)
    
    
    
    @card
    @catch
    @retry
    @step
    def logistic_regression(self):
        """  Logistic Regression Model Training """
        
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        self.model = LogisticRegression()
            
        self.target =  (self.preparedData["restecg"] == 'normal').astype(int)
        x_train, x_test, target_train, target_test = train_test_split(self.X, self.target,test_size=0.3,random_state=10)
        
        self.model.fit(x_train, target_train)
        target_pred = self.model.predict(x_test)
        
        self.Accuracy = accuracy_score(target_pred, target_test)   
        self.next(self.join)
    
    @card
    @catch
    @retry
    @step
    def decision_tree(self):
        """  Decision Tree Model Training """
        
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.Y_train)
        Y_pred = self.model.predict(self.X_test)
        self.Accuracy = accuracy_score(Y_pred, self.Y_test)
        
        self.next(self.join)
        
    @card
    @catch
    @retry
    @step
    def random_forest(self):
        """  Random Forest Model Training """
        
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        self.model = RandomForestClassifier(n_estimators = 100)
        self.model.fit(self.X_train, self.Y_train)
        Y_pred = self.model.predict(self.X_test)
        self.Accuracy = accuracy_score(self.Y_test, Y_pred)
        
        self.dataset = self.dataset
        self.dataset_shape = self.dataset_shape
        
        self.next(self.join)
        
    
    
    
    @card
    @step
    def join(self, inputs):
        """Joining all Trained Model Metrices"""
        
        import pandas as pd
        
        models = ["Logistic Regression", "Decision Tree", "Random Forest"]
        self.accuracies = [inputs.logistic_regression.Accuracy, inputs.decision_tree.Accuracy, inputs.random_forest.Accuracy]
        
        
        
        self.accuracy_Data = {"models": models, "Accuracies": self.accuracies}
        self.accuracy_DF = pd.DataFrame(self.accuracy_Data)
        print(self.accuracy_DF)
        
        self.dataset = inputs.random_forest.dataset
        self.dataset_shape = inputs.random_forest.dataset_shape
        
        self.randomforest_model = inputs.random_forest.model
        self.decisiontree_model = inputs.decision_tree.model
        self.logisticregression_model = inputs.logistic_regression.model
        
        self.next(self.tracking_using_mlflow)
    
    
    @card
    @step
    def tracking_using_mlflow(self):
        """Tracking the Project using MLflow, Logging params and metrices, for each Run. Saving each model."""
        # from sklearn.linear_model import LogisticRegression
        
        models = ["Logistic Regression", "Decision Tree", "Random Forest"]
        random_forest_model = self.randomforest_model
        ds_tree_model = self.decisiontree_model
        logreg_model = self.logisticregression_model
        
        mlflow.set_experiment("Disease Classifier using MLOps")
        with mlflow.start_run(run_name=f"Project Version-V1") as run: 
            mlflow.log_params({"model":str(models)})
            mlflow.log_params({"Dataset": str(self.dataset)})
            mlflow.log_params({"Dataset_Shape": str(self.dataset_shape)})
            
            mlflow.sklearn.log_model(random_forest_model, "Random Forest") 
            mlflow.sklearn.log_model(ds_tree_model, "Decision Tree") 
            mlflow.sklearn.log_model(logreg_model, "Logistic Regression") 
            
            
            Accuracy_Dict = {}       
            for index, accuracy in enumerate(self.accuracies):
                Accuracy_Dict["Accuracy_"+ models[index]] = accuracy

            mlflow.log_metrics(Accuracy_Dict)
            # mlflow.log_metrics(self.accuracy_Data)
        
        
        self.next(self.end)    
 
    @card
    @step
    def end(self):
        print("Execution Completed Successfully.")



def show_matrix(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    dataframe = copy.deepcopy(df)
    
    dataframe.drop("dataset", axis=1, inplace=True)
    dataframe.drop("restecg", axis=1, inplace=True)
    plt.figure(figsize=(10, 8))
    matrix = dataframe.corr()
    sns.heatmap(matrix, annot= True, cmap="coolwarm", fmt=".2f")
    return plt.gcf()    



if __name__ == "__main__":
    DiseaseClassifier()
