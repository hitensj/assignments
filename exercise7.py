import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('indexData.csv')

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna()  # Drop rows with missing values for simplicity

# Label encode the 'Index' column for compatibility with numeric models
encoder = LabelEncoder()
df['Index'] = encoder.fit_transform(df['Index'])

# Define features and target variables
features = ['Index', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
X = df[features]
y_close = df['Close']  # Target for regression
y_volume_class = (df['Volume'] > df['Volume'].median()).astype(int)  # Target for classification

# Train-test split
X_train, X_test, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.2, random_state=42)
_, _, y_volume_train, y_volume_test = train_test_split(X, y_volume_class, test_size=0.2, random_state=42)

# 1. Optimize KNN Regression
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

knn_params = {'knn__n_neighbors': [3, 5, 7, 9, 11]}
knn_grid = GridSearchCV(knn_pipeline, knn_params, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_close_train)

best_knn = knn_grid.best_estimator_
knn_optimized_preds = best_knn.predict(X_test)
knn_optimized_mse = mean_squared_error(y_close_test, knn_optimized_preds)

# 2. Optimize Linear Regression
lin_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

lin_reg_pipeline.fit(X_train, y_close_train)
lin_reg_optimized_preds = lin_reg_pipeline.predict(X_test)
lin_reg_optimized_mse = mean_squared_error(y_close_test, lin_reg_optimized_preds)

# 3. Optimize Logistic Regression
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=1000))
])

log_reg_params = {'log_reg__C': [0.01, 0.1, 1, 10, 100], 'log_reg__penalty': ['l2']}
log_reg_grid = GridSearchCV(log_reg_pipeline, log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_volume_train)

best_log_reg = log_reg_grid.best_estimator_
log_reg_optimized_preds = best_log_reg.predict(X_test)
log_reg_optimized_acc = accuracy_score(y_volume_test, log_reg_optimized_preds)

# Visualize results
# 1. Scatter plot for KNN Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_close_test, knn_optimized_preds, alpha=0.5, label="Predicted vs Actual (KNN)")
plt.plot([min(y_close_test), max(y_close_test)], [min(y_close_test), max(y_close_test)], 'r--', label="Ideal Fit")
plt.title("KNN Regression: Predicted vs Actual")
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.legend()
plt.grid()
plt.show()

# 2. Scatter plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_close_test, lin_reg_optimized_preds, alpha=0.5, label="Predicted vs Actual (Linear Regression)")
plt.plot([min(y_close_test), max(y_close_test)], [min(y_close_test), max(y_close_test)], 'r--', label="Ideal Fit")
plt.title("Linear Regression: Predicted vs Actual")
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.legend()
plt.grid()
plt.show()

# 3. Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_volume_test, log_reg_optimized_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Volume", "High Volume"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# Output optimized results
print("KNN Optimized MSE:", knn_optimized_mse)
print("Linear Regression Optimized MSE:", lin_reg_optimized_mse)
print("Logistic Regression Optimized Accuracy:", log_reg_optimized_acc)
