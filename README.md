# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1: Read the given Data.

STEP 2: Clean the Data Set using Data Cleaning Process.

STEP 3: Apply Feature Scaling for the feature in the data set.

STEP 4: Apply Feature Selection for the feature in the data set.

STEP 5: Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv('income(1) (1).csv')
df
```

<img width="1687" height="725" alt="image" src="https://github.com/user-attachments/assets/79d5a4b9-9ad8-4903-a9d4-05128305b07b" />

```
df.shape
```

<img width="131" height="33" alt="image" src="https://github.com/user-attachments/assets/7967bc15-5a17-475b-8aa2-20a2b8007b4b" />

```
from sklearn.preprocessing import LabelEncoder
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
X = df_encoded.drop('SalStat', axis=1)
y = df_encoded['SalStat']
```
```
print(X)
```

<img width="758" height="591" alt="image" src="https://github.com/user-attachments/assets/72caf687-1c40-4f11-85ec-4e74dd536ba0" />

```
print(y)
```

<img width="430" height="268" alt="image" src="https://github.com/user-attachments/assets/2a49b513-ba9f-4bb2-9dae-21a4d1ace0f7" />

```
from sklearn.feature_selection import SelectKBest, chi2
chi2_selector = SelectKBest(chi2, k=5)
chi2_selector.fit(X, y)

selected_features_chi2 = X.columns[chi2_selector.get_support()]
print("Selected Features (Chi-Square):", list(selected_features_chi2))

mi_score = pd.Series(chi2_selector.scores_, index=X.columns)
print(mi_score.sort_values(ascending = False))
```

<img width="991" height="313" alt="image" src="https://github.com/user-attachments/assets/3509b17c-f7e1-41ff-92c5-95c2a38a405f" />

```
from sklearn.feature_selection import f_classif
anova_selector = SelectKBest(f_classif, k=5)
anova_selector.fit(X, y)

selected_features_anova = X.columns[anova_selector.get_support()]
print("Selected Features (ANOVA F-test):", list(selected_features_anova))

mi_score = pd.Series(anova_selector.scores_, index=X.columns)
print(mi_score.sort_values(ascending = False))
```

<img width="956" height="306" alt="image" src="https://github.com/user-attachments/assets/396bd838-43b0-404b-9b08-6975297bf1c0" />

```
from sklearn.feature_selection import mutual_info_classif
mi_selector = SelectKBest(mutual_info_classif, k=5)
mi_selector.fit(X, y)

selected_features_mi = X.columns[mi_selector.get_support()]
print("Selected Features (Mutual Info):", list(selected_features_mi))

mi_score = pd.Series(mi_selector.scores_, index=X.columns)
print("\nMutual Information Scores:\n", mi_score.sort_values(ascending = False))
```

<img width="1021" height="355" alt="image" src="https://github.com/user-attachments/assets/732039ed-d25c-496c-a6bf-92129c9e54bb" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

selected_features_rfe = X.columns[rfe.support_]
print("Selected Features (RFE):", list(selected_features_rfe))
```

<img width="846" height="143" alt="image" src="https://github.com/user-attachments/assets/c8d54faa-f54b-40de-8ca1-b7a62496e2dd" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS

model = LogisticRegression(max_iter=100)
sfs = SFS(model, n_features_to_select=5)
sfs.fit(X, y)

selected_features_sfs = X.columns[sfs.support_]
print("Selected Features (SFS):", list(selected_features_sfs))
```

<img width="977" height="144" alt="image" src="https://github.com/user-attachments/assets/6fc74bac-a55e-44ab-8ed4-1f4a96dc1d70" />

```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print("Top 5 features (Random Forest Importance):", list(selected_features_rf))
```

<img width="1043" height="30" alt="image" src="https://github.com/user-attachments/assets/19d80b44-6c48-47e1-bdd0-56640769f3d0" />

```
from sklearn.linear_model import LassoCV
import numpy as np

lasso = LassoCV(cv=5).fit(X, y)
importance = np.abs(lasso.coef_)

selected_features_lasso = X.columns[importance > 0]
print("Selected Features (Lasso):", list(selected_features_lasso))
```

<img width="781" height="33" alt="image" src="https://github.com/user-attachments/assets/06f8e791-83f6-45ac-871c-c47bd56ad178" />

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv('income(1) (1).csv')
le = LabelEncoder()
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
X = df_encoded.drop('SalStat', axis=1)
y = df_encoded['SalStat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

<img width="534" height="337" alt="image" src="https://github.com/user-attachments/assets/8f624c18-d65e-49f5-a581-864f647eb00f" />

# RESULT:
Thus, the Feature selection and Feature scaling has been used on the given dataset and executed successfully.
