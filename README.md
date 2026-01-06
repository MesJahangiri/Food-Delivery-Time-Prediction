# Food-Delivery-Time-Prediction
Food delivery time prediction using machine learning models and real-world delivery data

## Overview
This project presents an end-to-end machine learning pipeline for predicting food delivery time using real-world delivery data. The goal is to estimate the total delivery duration based on order details, restaurant characteristics, customer location, and preparation time.

The project covers the full data science workflow, including data cleaning, feature engineering, preprocessing, model training, and evaluation. Multiple regression and ensemble models are implemented and compared to identify the most effective approach for this prediction task.

## Dataset

The dataset used in this project contains real-world food delivery records, including information about orders, restaurants, delivery personnel, customer locations, and delivery times.

Key features include:

- `Delivery_person_Age` & `Delivery_person_Ratings`: Demographics and performance of delivery personnel
- `Restaurant Latitude` & `Longitude`, `Delivery Location Latitude` & `Longitude`: Locations to calculate distance
- `Order_Date`, `Time_Orderd`, `Time_Order_picked`: Order and pickup timestamps
- `Weatherconditions`, `Road_traffic_density`, `Vehicle_condition`, `Type_of_order`, `Type_of_vehicle`, `Multiple_deliveries`, `Festival`, `City`: Contextual and categorical features
- `Time_taken(min)`: Target variable representing delivery time in minutes

The dataset contains **45,593 rows** and **20 columns**. After cleaning and preprocessing, **37,918 rows** remained for modeling.

## Preprocessing & Feature Engineering

The raw dataset contained missing values, inconsistent formats, and outliers. Comprehensive data cleaning was performed to:

- Handle missing values (`NaN`) in numeric and categorical columns
- Normalize and standardize timestamps and categorical strings
- Remove unrealistic location coordinates and delivery distances

Feature engineering included:

- Calculating `distance_km` between restaurants and delivery locations using Haversine formula
- Extracting order day of the week, order hour, and pickup hour as numeric features
- Computing `prep_time_minutes` as the difference between order and pickup time
- Encoding categorical variables for modeling

