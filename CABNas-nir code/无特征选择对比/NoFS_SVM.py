import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.signal import savgol_filter

def process_spectrum_data(file_path):
    """
    不使用特征选择的SVM分类
    """
    print("="*60)
    print("           无特征选择 - SVM分类")
    print("="*60)
    
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    print(f"原始数据形状: {X.shape}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 2. SG预处理
    print("正在进行SG预处理...")
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 3. 数据划分
    print("正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y, test_size=0.4, random_state=44,
    )

    # 4. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"使用全部特征数量: {X_train_scaled.shape[1]}")

    # 5. 使用GridSearchCV进行SVM模型训练和参数优化（使用全部特征）
    print("正在进行SVM训练和参数优化（使用全部特征）...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 6. 模型评估
    print("正在进行模型评估...")
    y_pred = best_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- 模型评估结果 (SVM - 无特征选择) ---")
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    
    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 显示特征使用信息
    print(f"\n特征使用信息:")
    print(f"使用特征数量: {X_train_scaled.shape[1]} (全部特征)")
    print(f"特征选择方法: 无")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_features': X_train_scaled.shape[1],
        'method': 'SVM-无特征选择'
    }

if __name__ == '__main__':
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    results = process_spectrum_data(data_path) 