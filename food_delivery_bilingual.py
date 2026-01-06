# Food Delivery Time Prediction - Bilingual Inline Notebook
# -----------------------------------------------
# All comments: Persian + English inline

# 1️⃣ Imports و بارگذاری داده / Imports and data loading
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Load dataset / بارگذاری داده
df = pd.read_csv("fooddata.csv") 
print(df.head())  # نمایش پنج ردیف اول / Display first 5 rows
print(df.info())  # اطلاعات کلی دیتافریم / General dataframe info

# Create a copy for cleaning / ایجاد نسخه‌ای برای تمیزکاری
df_clean = df.copy()

# -----------------------------------------------
# 2️⃣ Preprocessing ستون‌ها / Column preprocessing
# Cleaning ID / تمیزکاری ستون ID
df_clean["ID"] = df_clean["ID"].str.strip()  # حذف فاصله‌ها / Strip spaces
df_clean.replace({"ID": {"": pd.NA, "nan": pd.NA, "NaN": pd.NA}}, inplace=True)  # توکن‌های گم شده / Replace missing tokens
df_clean["ID"] = df_clean["ID"].astype("category")  # تبدیل به category / Convert to category

# Cleaning Delivery Person ID / تمیزکاری Delivery_person_ID
df_clean["Delivery_person_ID"] = df_clean["Delivery_person_ID"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
df_clean["Delivery_person_ID"] = df_clean["Delivery_person_ID"].astype("category")

# Cleaning Delivery Person Age / تمیزکاری Delivery_person_Age
age_series = df_clean["Delivery_person_Age"].astype(str).str.strip().replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA})
df_clean["Delivery_person_Age"] = pd.to_numeric(age_series, errors="coerce").astype("Int64")

# Cleaning Delivery Person Ratings / تمیزکاری Delivery_person_Ratings
ratings_series = df_clean["Delivery_person_Ratings"].astype(str).str.strip().replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA})
df_clean["Delivery_person_Ratings"] = pd.to_numeric(ratings_series, errors="coerce").astype("Float64")

# Cleaning Order Date / تمیزکاری Order_Date
order_date_raw = df_clean["Order_Date"].astype(str).str.strip().replace({"": pd.NA, "NaN": pd.NA, "nan": pd.NA})
order_date_cleaned = order_date_raw.str.replace(r"[./]", "-", regex=True).str.replace(r"\s*--\s*", "-", regex=True).str.replace(r"\s+", "", regex=True)
df_clean["Order_Date"] = pd.to_datetime(order_date_cleaned, dayfirst=True, errors="coerce")

# Cleaning Time Ordered / تمیزکاری Time_Orderd
time_raw = df_clean["Time_Orderd"].astype(str).str.strip().replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA})
df_clean["Time_Orderd"] = pd.to_datetime(time_raw, format="%H:%M:%S", errors="coerce")

# Cleaning Time Order Picked / تمیزکاری Time_Order_picked
time_raw = df_clean["Time_Order_picked"].astype(str).str.strip().replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA})
df_clean["Time_Order_picked"] = pd.to_datetime(time_raw, format="%H:%M:%S", errors="coerce")

# Cleaning categorical features (Weather, Traffic, Type) / تمیزکاری ستون‌های دسته‌ای
for col in ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", "City"]:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower().replace({"": pd.NA, "nan": pd.NA}).astype('category')

# Cleaning numerical categorical features (Vehicle_condition, Multiple Deliveries, Festival) / تمیزکاری ستون‌های عددی
for col in ["Vehicle_condition", "multiple_deliveries", "Festival"]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')

# Cleaning Time_taken(min) / تمیزکاری Time_taken(min)
tt_series = df_clean["Time_taken(min)"].astype(str).str.strip().replace({"nan": pd.NA, "NaN": pd.NA, "" : pd.NA})
tt_series = tt_series.str.replace(r"\(min\)\s*", "", regex=True)
df_clean["Time_taken(min)"] = pd.to_numeric(tt_series, errors="coerce").astype("Int64")

# Remove rows with any missing data / حذف ردیف‌های دارای داده گم‌شده
df_ready = df_clean.dropna().reset_index(drop=True)

# -----------------------------------------------
# 3️⃣ Feature Engineering / مهندسی ویژگی
# Haversine distance / محاسبه فاصله بین رستوران و محل تحویل

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km / شعاع زمین به کیلومتر
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df_ready["distance_km"] = haversine(df_ready["Restaurant_latitude"], df_ready["Restaurant_longitude"], df_ready["Delivery_location_latitude"], df_ready["Delivery_location_longitude"])

# Convert times to numeric / تبدیل زمان‌ها به عدد
 df_ready["order_dayofweek"] = df_ready["Order_Date"].dt.weekday.astype("int8")
df_ready["order_hour_float"] = (df_ready["Time_Orderd"].dt.hour + df_ready["Time_Orderd"].dt.minute / 60).astype("float32")
df_ready["pickup_hour_float"] = (df_ready["Time_Order_picked"].dt.hour + df_ready["Time_Order_picked"].dt.minute / 60).astype("float32")
df_ready["prep_time_minutes"] = ((df_ready["pickup_hour_float"] - df_ready["order_hour_float"]).where(lambda x: x >=0, lambda x: x + 24) * 60).astype("float32")

# Drop original columns after feature engineering / حذف ستون‌های اولیه
cols_to_drop = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude", "Time_Orderd", "Time_Order_picked", "Order_Date"]
df_ready_final = df_ready.drop(columns=cols_to_drop)

# -----------------------------------------------
# 4️⃣ Modeling / مدل سازی
numeric_cols = ["Delivery_person_Age", "Delivery_person_Ratings", "distance_km", "order_hour_float", "pickup_hour_float", "prep_time_minutes"]
categorical_cols = ["Vehicle_condition", "multiple_deliveries", "Festival", "order_dayofweek", "Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", "City"]
preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)])

X = df_ready_final.drop(columns=["Time_taken(min)", "ID", "Delivery_person_ID"])
y = df_ready_final["Time_taken(min)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Linear Regression / رگرسیون خطی
model_lr = LinearRegression()
model_lr.fit(X_train_transformed, y_train)
y_pred_lr = model_lr.predict(X_test_transformed)
print("MAE_LinearRegression:", round(mean_absolute_error(y_test, y_pred_lr),2))
print("RMSE_LinearRegression:", round(np.sqrt(mean_squared_error(y_test, y_pred_lr)),2))

# RandomForest / رندوم فارست
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_transformed, y_train)
y_pred_rf = model_rf.predict(X_test_transformed)
print("MAE_RandomForest:", round(mean_absolute_error(y_test, y_pred_rf),2))
print("RMSE_RandomForest:", round(np.sqrt(mean_squared_error(y_test, y_pred_rf)),2))

# Gradient Boosting / گرادیان بوستینگ
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb.fit(X_train_transformed, y_train)
y_pred_gb = model_gb.predict(X_test_transformed)
print("MAE_GradientBoosting:", round(mean_absolute_error(y_test, y_pred_gb),2))
print("RMSE_GradientBoosting:", round(np.sqrt(mean_squared_error(y_test, y_pred_gb)),2))

# Feature Importance / اهمیت ویژگی‌ها
importances = model_rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
print(feat_imp_df.head(10))
