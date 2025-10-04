<img width="1326" height="684" alt="image" src="https://github.com/user-attachments/assets/820346e8-53a5-4ac4-9625-7433a99a112f" />


📊 Advanced CSV Data Analyzer

An interactive and feature-rich Streamlit web application that allows users to upload, clean, analyze, visualize, and perform PCA (Principal Component Analysis) on CSV datasets — all in one place.

This app helps data scientists, analysts, and learners quickly explore datasets without writing a single line of code.

🚀 Features

🔍 1. Dataset Overview

View the first few rows of your dataset.

Display shape, column names, and data types.

Generate descriptive statistics for selected columns.

Detect and display missing values.

Download the processed dataset as a CSV file.

🧼 2. Data Cleaning & Transformation

Drop unwanted columns interactively.

Fill missing values using:

Mean, Median, Mode

Custom user-defined values

Convert data types (int, float, str, category).

Filter rows using:

Numeric range sliders

Categorical multi-select filters

Reset to original dataset anytime.

📈 3. Exploratory Data Analysis

Value counts for any column.

Correlation matrix with heatmap visualization.

Top correlated pairs display.

Column-level statistics:

Numeric: Histogram, Box plot, Violin plot

Categorical: Bar chart, Pie chart

📊 4. Data Visualization

Generate different charts dynamically:

Histogram, Box Plot, Scatter Plot, Line Plot, Bar Chart

Customize axes, color, and size encodings.

3D Scatter Plot for multi-variable visual exploration.

🤖 5. PCA (Principal Component Analysis)

Perform dimensionality reduction on numeric columns.

Optionally standardize data for better results.

Choose number of principal components.

Visualize PCA output in:

2D or 3D Scatter Plots

View explained variance ratio and component loadings.

🛠️ Tech Stack
Component	Technology Used
Frontend/UI	Streamlit

Data Handling	Pandas
, NumPy

Visualization	Plotly
, Seaborn
, Matplotlib

Machine Learning	Scikit-learn (PCA, StandardScaler)

Styling	Custom CSS for Streamlit
