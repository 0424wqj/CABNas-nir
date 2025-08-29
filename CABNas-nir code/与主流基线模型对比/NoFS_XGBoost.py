import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.signal import savgol_filter

def process_spectrum_data(file_path):
    """
    不使用特征选择的XGBoost分类
    """
    print("="*60)
    print("           无特征选择 - XGBoost分类")
    print("="*60)
    
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    print(f"原始数据形状: {X.shape}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 2. 标签编码（XGBoost要求）
    print("正在进行标签编码...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"编码后标签范围: {np.min(y_encoded)} - {np.max(y_encoded)}")
    print(f"类别数量: {len(le.classes_)}")

    # 3. SG预处理
    print("正在进行SG预处理...")
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 4. 数据划分
    print("正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # 5. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"使用全部特征数量: {X_train_scaled.shape[1]}")

    # 6. 使用GridSearchCV进行XGBoost训练和参数优化（使用全部特征）
    print("正在进行XGBoost训练和参数优化（使用全部特征）...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), 
        param_grid, 
        cv=3, 
        n_jobs=-1,
        scoring='accuracy'
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 7. 模型评估
    print("正在进行模型评估...")
    y_pred = best_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- 模型评估结果 (XGBoost - 无特征选择) ---")
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    


if __name__ == '__main__':
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    results = process_spectrum_data(data_path) 