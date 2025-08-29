import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
        
        # 存储每一次迭代选择的特征索引和对应的RMSECV
        retained_feature_indices_history = []

        # 初始特征集包含所有特征
        retained_indices = list(range(n_features))

        for i in range(self.n_iterations):
            # 1. 在当前特征子集上建立PLS模型
            pls = PLSRegression(n_components=min(10, len(retained_indices), n_samples - 1))
            pls.fit(X[:, retained_indices], y)

            # 2. 计算每个特征的权重（回归系数的绝对值）
            weights = np.abs(pls.coef_).flatten()
            
            # 3. 确定本次迭代要保留的特征数量 (指数衰减)
            ratio = (1 / (i + 1)) ** 0.3
            n_retained = max(2, int(n_features * ratio))
            
            # 4. 根据权重排序，保留权重最高的特征
            sorted_indices = np.argsort(weights)[::-1]
            retained_indices_local = sorted_indices[:n_retained]
            # 将局部索引映射回原始索引
            retained_indices = np.array(retained_indices)[retained_indices_local]
            retained_feature_indices_history.append(retained_indices)

            if len(retained_indices) < 2:
                break

        # 5. 交叉验证评估所有历史特征子集
        print("Evaluating feature subsets with cross-validation...")
        for i, indices in enumerate(retained_feature_indices_history):
            if len(indices) < 2:
                continue
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
    # 1. 加载数据，与 20分类.py 保持一致
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # 2. SG预处理
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 3. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y, test_size=0.3, random_state=45, stratify=y
    )

    # 重新启用标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. CARS特征选择
    print("Starting CARS feature selection...")
    cars = CARS(n_iterations=50, cv_folds=5)
    cars.fit(X_train_scaled, y_train)
    
    X_train_cars = cars.transform(X_train_scaled)
    X_test_cars = cars.transform(X_test_scaled)
    print(f"CARS selected {X_train_cars.shape[1]} features.")

    # 5. 使用GridSearchCV进行SVM模型训练和参数优化 (使用CARS选择的特征)
    print("Starting SVM training with GridSearchCV on CARS selected features...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_cars, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best model found: {best_model}")

    # 6. 模型评估
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