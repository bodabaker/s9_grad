import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_model(n_estimators, random_state):

    data = pd.read_csv("Cleaned Amazon Sale Report.csv")
    y = data['Amount']
    x = data.drop('Amount', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train = pd.get_dummies(x_train) 

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, pred)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    with mlflow.start_run(nested=True):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("R2 Score", r2)

        mlflow.sklearn.log_model(model, "Random Forest Regressor")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    train_model(args.n_estimators, args.random_state)