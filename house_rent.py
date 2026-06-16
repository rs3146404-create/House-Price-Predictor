#%%
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%%
data = pd.read_csv('House_Rent_Dataset.csv')
df = pd.DataFrame(data)
df.head()
#%%
df = df.drop(columns=['Posted On','Area Locality','Point of Contact'])
df.head()
df['Bathroom'].value_counts()
#%%
num_cols = ['BHK', 'Size', 'Bathroom']

cat_cols = [
    'Floor',
    'Area Type',
    'City',
    'Furnishing Status',
    'Tenant Preferred',
]
#%%
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
#%%
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
#%%
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
#%%

model = Pipeline([
 ('preprocessor', preprocessor),
 ('regressor',LinearRegression())
])
#%%
X = df[num_cols + cat_cols]  # ✅ fix
# ✅ Step 1: outlier হটা
X = df[num_cols + cat_cols]
df_clean = df[df['Rent'] <= df['Rent'].quantile(0.99)].copy().reset_index(drop=True)  # ✅ reset_index

y = np.log1p(df_clean['Rent'])  # ✅ df_clean থেকে নে
X = df_clean[num_cols + cat_cols]  # ✅ X ও df_clean থেকে

#%%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#%%
model.fit(X_train, y_train)

#%%
y_pred = model.predict(X_test)
#%%
mae = mean_absolute_error(y_test, y_pred)
print('MAE:', mae)
#%%
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
#%%
rmse = np.sqrt(mse)
print('RMSE:', rmse)
#%%
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)
#%%


joblib.dump(model,'house_rent_model.pkl')

#%%
model = joblib.load('house_rent_model.pkl')
# model