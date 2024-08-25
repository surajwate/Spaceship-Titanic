import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def run(fold):
    # Load the full training data with folds
    df = pd.read_csv("./input/train_folds.csv")

    # Drop the PassengerId column
    df.drop(columns=['PassengerId'], inplace=True)

    # Create a new columns from the split values of the Cabin column
    df[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = df.Cabin.str.split("/", expand=True)
    df.drop(columns=['Cabin'], inplace=True)

    # Convert the Number column to numeric
    df['Cabin_Number'] = pd.to_numeric(df['Cabin_Number'], errors='coerce')

    # Split the data into training and validation
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the kfold column
    df_train = df_train.drop(columns=['kfold'])
    df_valid = df_valid.drop(columns=['kfold'])

    # Numerical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols.remove('kfold')

    # Separate the features and target
    X_train = df_train.drop('Transported', axis=1)
    X_valid = df_valid.drop('Transported', axis=1)
    y_train = df_train.Transported
    y_valid = df_valid.Transported

    # Impute missing values
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_valid[num_cols] = num_imputer.transform(X_valid[num_cols])
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_valid[cat_cols] = cat_imputer.transform(X_valid[cat_cols])

    # Scale the numerical columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid[num_cols] = scaler.transform(X_valid[num_cols])

    # One-hot encode the categorical columns
    ohe = OneHotEncoder()

    # Fit ohe on training + validation features
    full_data = pd.concat(
        [X_train[cat_cols], X_valid[cat_cols]],
        axis=0
    )
    ohe.fit(full_data)

    # Transform training and validation features
    X_train_ohe = ohe.transform(X_train[cat_cols]).toarray()
    X_valid_ohe = ohe.transform(X_valid[cat_cols]).toarray()

    # Create final training and validation datasets
    X_train = np.hstack((X_train[num_cols], X_train_ohe))
    X_valid = np.hstack((X_valid[num_cols], X_valid_ohe))

    # Initialize the model
    model = LogisticRegression()

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Predict on the validation data
    preds = model.predict(X_valid)

    # Calculate the accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
