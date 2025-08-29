import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter
import xgboost as xgb
import random

class Particle:
    """粒子类，表示一个特征选择方案"""
    def __init__(self, n_features):
        # 随机初始化粒子位置（二进制向量，表示特征选择）
        self.position = np.random.randint(0, 2, n_features)
        # 确保至少选择2个特征
        if np.sum(self.position) < 2:
            indices = np.random.choice(n_features, 2, replace=False)
            self.position = np.zeros(n_features)
            self.position[indices] = 1
        
        # 初始化速度
        self.velocity = np.random.uniform(-1, 1, n_features)
        
        # 个体最佳位置和适应度
        self.best_position = self.position.copy()
        self.best_fitness = -float('inf')
        
        # 当前适应度
        self.fitness = -float('inf')

class PSO:
    """粒子群优化特征选择算法"""
    def __init__(self, n_particles=30, n_iterations=50, w=0.5, c1=2.0, c2=2.0, cv_folds=5):
        self.n_particles = n_particles  # 粒子数量
        self.n_iterations = n_iterations  # 迭代次数
        self.w = w  # 惯性权重
        self.c1 = c1  # 认知系数
        self.c2 = c2  # 社会系数
        self.cv_folds = cv_folds  # 交叉验证折数
        self.best_global_position = None
        self.best_global_fitness = -float('inf')
        self.particles = []

    def fitness_function(self, X, y, feature_mask):
        """适应度函数：基于选择的特征进行XGBoost交叉验证"""
        if np.sum(feature_mask) < 2:  # 至少需要2个特征
            return -float('inf')
        
        selected_features = np.where(feature_mask == 1)[0]
        X_selected = X[:, selected_features]
        
        # 使用交叉验证评估特征子集
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        accuracies = []
        
        for train_idx, val_idx in kf.split(X_selected):
            X_train_cv, X_val_cv = X_selected[train_idx], X_selected[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # XGBoost模型
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='mlogloss'  # 多分类损失函数
            )
            xgb_model.fit(X_train_cv, y_train_cv)
            y_pred_cv = xgb_model.predict(X_val_cv)
            accuracy = accuracy_score(y_val_cv, y_pred_cv)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        # 添加特征数量惩罚项，鼓励选择较少的特征
        penalty = len(selected_features) / X.shape[1] * 0.01
        return avg_accuracy - penalty

    def update_velocity(self, particle, global_best_position):
        """更新粒子速度"""
        r1 = np.random.random(len(particle.velocity))
        r2 = np.random.random(len(particle.velocity))
        
        cognitive_component = self.c1 * r1 * (particle.best_position - particle.position)
        social_component = self.c2 * r2 * (global_best_position - particle.position)
        
        particle.velocity = (self.w * particle.velocity + 
                           cognitive_component + 
                           social_component)
        
        # 限制速度范围
        particle.velocity = np.clip(particle.velocity, -6, 6)

    def update_position(self, particle):
        """更新粒子位置"""
        # 使用sigmoid函数将速度转换为概率
        sigmoid_velocity = 1 / (1 + np.exp(-particle.velocity))
        
        # 根据概率更新位置
        random_values = np.random.random(len(particle.position))
        particle.position = (random_values < sigmoid_velocity).astype(int)
        
        # 确保至少选择2个特征
        if np.sum(particle.position) < 2:
            indices = np.random.choice(len(particle.position), 2, replace=False)
            particle.position = np.zeros(len(particle.position))
            particle.position[indices] = 1

    def fit(self, X, y):
        """训练PSO特征选择算法"""
        n_features = X.shape[1]
        
        # 初始化粒子群
        self.particles = [Particle(n_features) for _ in range(self.n_particles)]
        
        print(f"开始粒子群优化特征选择，粒子数: {self.n_particles}, 迭代次数: {self.n_iterations}")
        
        for iteration in range(self.n_iterations):
            print(f"迭代 {iteration + 1}/{self.n_iterations}")
            
            # 计算每个粒子的适应度
            for particle in self.particles:
                particle.fitness = self.fitness_function(X, y, particle.position)
                
                # 更新个体最佳
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # 更新全局最佳
                if particle.fitness > self.best_global_fitness:
                    self.best_global_fitness = particle.fitness
                    self.best_global_position = particle.position.copy()
            
            # 更新粒子速度和位置
            for particle in self.particles:
                self.update_velocity(particle, self.best_global_position)
                self.update_position(particle)
            
            if iteration % 10 == 0:
                n_selected = np.sum(self.best_global_position)
                print(f"  当前最佳适应度: {self.best_global_fitness:.4f}, 选择特征数: {n_selected}")
        
        print(f"PSO优化完成，最终选择 {np.sum(self.best_global_position)} 个特征")
        return self

    def transform(self, X):
        """使用最佳特征子集转换数据"""
        if self.best_global_position is None:
            raise ValueError("请先调用fit方法训练模型")
        
        selected_features = np.where(self.best_global_position == 1)[0]
        return X[:, selected_features]

def process_spectrum_data(file_path):
    # 1. 加载数据，与 20分类.py 保持一致
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # 2. 标签编码（XGBoost需要0开始的标签）
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. SG预处理
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 4. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # 5. 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. PSO特征选择
    print("开始粒子群算法特征选择...")
    pso = PSO(n_particles=30, n_iterations=10, w=0.5, c1=2.0, c2=2.0, cv_folds=5)
    pso.fit(X_train_scaled, y_train)
    
    X_train_pso = pso.transform(X_train_scaled)
    X_test_pso = pso.transform(X_test_scaled)
    print(f"PSO选择了 {X_train_pso.shape[1]} 个特征")

    # 7. 使用GridSearchCV进行XGBoost模型训练和参数优化（使用PSO选择的特征）
    print("开始XGBoost训练和参数优化...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'), 
        param_grid, 
        cv=3, 
        n_jobs=-1
    )
    grid_search.fit(X_train_pso, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {best_model}")

    # 8. 模型评估
    y_pred = best_model.predict(X_test_pso)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- 模型评估结果 ---")
    print(f"准确率: {accuracy:.2f}")
    print(f"精准率: {precision:.2f}")
    print(f"召回率: {recall:.2f}")
    print(f"F1值: {f1:.2f}")
    
    # 显示选择的特征信息
    selected_features = np.where(pso.best_global_position == 1)[0]
    print(f"\n选择的特征索引: {selected_features}")
    print(f"特征选择率: {len(selected_features)}/{X.shape[1]} = {len(selected_features)/X.shape[1]*100:.2f}%")
    
    # 显示标签映射信息
    print(f"\n标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

if __name__ == '__main__':
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    process_spectrum_data(data_path) 