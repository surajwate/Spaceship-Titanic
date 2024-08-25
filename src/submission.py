import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Load the full training data with folds
df_train = pd.read_csv("./input/train.csv")
df_test = pd.read_csv("./input/test.csv")

# Drop the PassengerId column
df_train.drop(columns=['PassengerId'], inplace=True)
test_ids = df_test['PassengerId']
df_test.drop(columns=['PassengerId'], inplace=True)

# Create a new columns from the split values of the Cabin column
df_train[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = df_train.Cabin.str.split("/", expand=True)
df_train.drop(columns=['Cabin'], inplace=True)
df_test[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = df_test.Cabin.str.split("/", expand=True)
df_test.drop(columns=['Cabin'], inplace=True)

# Convert the Number column to numeric
df_train['Cabin_Number'] = pd.to_numeric(df_train['Cabin_Number'], errors='coerce')
df_test['Cabin_Number'] = pd.to_numeric(df_test['Cabin_Number'], errors='coerce')

# Numerical columns
num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = df_train.select_dtypes(include='object').columns.tolist()

# Separate the features and target
X_train = df_train.drop('Transported', axis=1)
y_train = df_train.Transported
X_test = df_test

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# Scale the numerical columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# One-hot encode the categorical columns
ohe = OneHotEncoder()

# Fit ohe on training + validation features
full_data = pd.concat(
    [X_train[cat_cols], X_test[cat_cols]],
    axis=0
)
ohe.fit(full_data)

# Transform training and validation features
X_train_ohe = ohe.transform(X_train[cat_cols]).toarray()
X_test_ohe = ohe.transform(X_test[cat_cols]).toarray()

# Create final training and validation datasets
X_train = np.hstack((X_train[num_cols], X_train_ohe))
X_test = np.hstack((X_test[num_cols], X_test_ohe))

# Initialize the model
model = LogisticRegression()

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on the validation data
preds = model.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": preds
})

submission.to_csv("./output/submission.csv", index=False)
print("Submission saved successfully!")
