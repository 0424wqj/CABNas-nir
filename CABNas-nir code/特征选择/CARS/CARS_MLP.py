import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

class CARS:
    def __init__(self, n_iterations=50, cv_folds=5):
        self.n_iterations = n_iterations
        self.cv_folds = cv_folds
        self.best_feature_indices_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.best_rmse_ = float('inf')
        
        retained_feature_indices_history = []
        retained_indices = list(range(n_features))

        for i in range(self.n_iterations):
            pls = PLSRegression(n_components=min(10, len(retained_indices), n_samples - 1))
            pls.fit(X[:, retained_indices], y)
            weights = np.abs(pls.coef_).flatten()
            ratio = (1 / (i + 1)) ** 0.5
            n_retained = max(2, int(n_features * ratio))
            sorted_indices = np.argsort(weights)[::-1]
            retained_indices_local = sorted_indices[:n_retained]
            retained_indices = np.array(retained_indices)[retained_indices_local]
            retained_feature_indices_history.append(retained_indices)
            if len(retained_indices) < 2:
                break

        print("Evaluating feature subsets with cross-validation...")
        for i, indices in enumerate(retained_feature_indices_history):
            if len(indices) < 2: continue
            X_subset = X[:, indices]
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            rmse_cv_scores = []
            for train_idx, val_idx in kf.split(X_subset):
                X_train_cv, X_val_cv = X_subset[train_idx], X_subset[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                pls_cv = PLSRegression(n_components=min(10, X_train_cv.shape[1], X_train_cv.shape[0] - 1))
                pls_cv.fit(X_train_cv, y_train_cv)
                y_pred_cv = pls_cv.predict(X_val_cv)
                rmse_cv_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))
            avg_rmse = np.mean(rmse_cv_scores)
            if avg_rmse < self.best_rmse_:
                self.best_rmse_ = avg_rmse
                self.best_feature_indices_ = indices
        
        if self.best_feature_indices_ is None and retained_feature_indices_history:
             self.best_feature_indices_ = retained_feature_indices_history[-1]
        return self

    def transform(self, X):
        return X[:, self.best_feature_indices_]

def process_spectrum_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y, test_size=0.3, random_state=48, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Starting CARS feature selection...")
    cars = CARS(n_iterations=50, cv_folds=5)
    cars.fit(X_train_scaled, y_train)
    
    X_train_cars = cars.transform(X_train_scaled)
    X_test_cars = cars.transform(X_test_scaled)
    print(f"CARS selected {X_train_cars.shape[1]} features.")

    print("Starting MLP training with GridSearchCV on CARS selected features...")
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001]
    }
    grid_search = GridSearchCV(MLPClassifier(random_state=42, max_iter=1000), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_cars, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best model found: {best_model}")

    y_pred = best_model.predict(X_test_cars)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("--- 模型评估结果 ---")
    print(f"准确率: {accuracy:.2f}")
    print(f"精准率: {precision:.2f}")
    print(f"召回率: {recall:.2f}")
    print(f"F1值: {f1:.2f}")

if __name__ == '__main__':
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    process_spectrum_data(data_path)