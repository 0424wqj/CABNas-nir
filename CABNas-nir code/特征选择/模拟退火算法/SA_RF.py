import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter
import random
import math

class Solution:
    """解类，表示一个特征选择方案"""
    def __init__(self, n_features):
        # 随机初始化特征选择方案（二进制向量）
        self.features = np.random.randint(0, 2, n_features)
        # 确保至少选择2个特征
        if np.sum(self.features) < 2:
            indices = np.random.choice(n_features, 2, replace=False)
            self.features = np.zeros(n_features)
            self.features[indices] = 1
        
        self.fitness = 0.0
        self.n_features = n_features

    def copy(self):
        """创建解的副本"""
        new_solution = Solution(self.n_features)
        new_solution.features = self.features.copy()
        new_solution.fitness = self.fitness
        return new_solution

    def get_neighbor(self):
        """生成邻域解：随机翻转1-3个特征的选择状态"""
        neighbor = self.copy()
        
        # 随机选择要翻转的特征数量（1-3个）
        n_flips = random.randint(1, min(3, self.n_features))
        
        # 随机选择要翻转的特征索引
        flip_indices = np.random.choice(self.n_features, n_flips, replace=False)
        
        # 翻转选择状态
        for idx in flip_indices:
            neighbor.features[idx] = 1 - neighbor.features[idx]
        
        # 确保至少选择2个特征
        if np.sum(neighbor.features) < 2:
            # 随机选择2个特征设为1
            indices = np.random.choice(self.n_features, 2, replace=False)
            neighbor.features = np.zeros(self.n_features)
            neighbor.features[indices] = 1
        
        return neighbor

    def get_selected_features(self):
        """获取选择的特征索引"""
        return np.where(self.features == 1)[0]

class SimulatedAnnealing:
    """模拟退火算法特征选择"""
    def __init__(self, initial_temperature=100, final_temperature=0.01, 
                 cooling_rate=0.95, max_iterations=1000, cv_folds=5):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.cv_folds = cv_folds
        self.best_solution = None
        self.best_fitness = 0.0
        self.temperature_history = []
        self.fitness_history = []
        self.acceptance_history = []

    def fitness_function(self, X, y, solution):
        """适应度函数：基于选择的特征进行随机森林交叉验证"""
        selected_features = solution.get_selected_features()
        
        if len(selected_features) < 2:
            return 0.0
        
        X_selected = X[:, selected_features]
        
        # 使用交叉验证评估特征子集
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        accuracies = []
        
        for train_idx, val_idx in kf.split(X_selected):
            X_train_cv, X_val_cv = X_selected[train_idx], X_selected[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # 随机森林模型
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_cv, y_train_cv)
            y_pred_cv = rf.predict(X_val_cv)
            accuracy = accuracy_score(y_val_cv, y_pred_cv)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        
        # 添加特征数量惩罚项，鼓励选择较少的特征
        feature_penalty = len(selected_features) / X.shape[1] * 0.01
        fitness = avg_accuracy - feature_penalty
        
        return fitness

    def accept_solution(self, current_fitness, new_fitness, temperature):
        """Metropolis接受准则"""
        if new_fitness > current_fitness:
            # 新解更好，直接接受
            return True
        else:
            # 新解较差，按概率接受
            delta = new_fitness - current_fitness
            probability = math.exp(delta / temperature)
            return random.random() < probability

    def cool_down(self, temperature):
        """冷却策略：指数衰减"""
        return temperature * self.cooling_rate

    def fit(self, X, y):
        """训练模拟退火特征选择算法"""
        n_features = X.shape[1]
        
        # 初始化当前解
        current_solution = Solution(n_features)
        current_solution.fitness = self.fitness_function(X, y, current_solution)
        
        # 初始化最佳解
        self.best_solution = current_solution.copy()
        self.best_fitness = current_solution.fitness
        
        # 初始化温度
        temperature = self.initial_temperature
        
        print(f"开始模拟退火特征选择")
        print(f"初始温度: {self.initial_temperature}, 最终温度: {self.final_temperature}")
        print(f"冷却率: {self.cooling_rate}, 最大迭代数: {self.max_iterations}")
        
        accepted_count = 0
        iteration = 0
        
        while temperature > self.final_temperature and iteration < self.max_iterations:
            iteration += 1
            
            # 生成邻域解
            neighbor_solution = current_solution.get_neighbor()
            neighbor_solution.fitness = self.fitness_function(X, y, neighbor_solution)
            
            # 决定是否接受新解
            accept = self.accept_solution(current_solution.fitness, 
                                        neighbor_solution.fitness, 
                                        temperature)
            
            if accept:
                current_solution = neighbor_solution
                accepted_count += 1
                
                # 更新最佳解
                if current_solution.fitness > self.best_fitness:
                    self.best_solution = current_solution.copy()
                    self.best_fitness = current_solution.fitness
            
            # 记录历史
            self.temperature_history.append(temperature)
            self.fitness_history.append(current_solution.fitness)
            self.acceptance_history.append(accept)
            
            # 降温
            temperature = self.cool_down(temperature)
            
            # 定期输出进度
            if iteration % 100 == 0:
                acceptance_rate = accepted_count / iteration * 100
                n_selected = np.sum(self.best_solution.features)
                print(f"迭代 {iteration:4d}: 温度={temperature:.4f}, "
                      f"当前适应度={current_solution.fitness:.4f}, "
                      f"最佳适应度={self.best_fitness:.4f}, "
                      f"接受率={acceptance_rate:.1f}%, "
                      f"最佳特征数={n_selected}")
        
        final_acceptance_rate = accepted_count / iteration * 100
        print(f"\n模拟退火优化完成！")
        print(f"总迭代次数: {iteration}")
        print(f"最终温度: {temperature:.6f}")
        print(f"总体接受率: {final_acceptance_rate:.2f}%")
        print(f"最佳适应度: {self.best_fitness:.4f}")
        print(f"最终选择 {np.sum(self.best_solution.features)} 个特征")
        
        return self

    def transform(self, X):
        """使用最佳解的特征选择方案转换数据"""
        if self.best_solution is None:
            raise ValueError("请先调用fit方法训练模型")
        
        selected_features = self.best_solution.get_selected_features()
        return X[:, selected_features]

    def get_selected_features(self):
        """获取选择的特征索引"""
        if self.best_solution is None:
            return None
        return self.best_solution.get_selected_features()

    def get_optimization_info(self):
        """获取优化过程信息"""
        return {
            'temperature_history': self.temperature_history,
            'fitness_history': self.fitness_history,
            'acceptance_history': self.acceptance_history,
            'final_acceptance_rate': sum(self.acceptance_history) / len(self.acceptance_history) * 100
        }

def process_spectrum_data(file_path):
    # 1. 加载数据，与原SA版本保持一致
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # 2. SG预处理
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 3. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y, test_size=0.3, random_state=43, stratify=y
    )

    # 4. 数据标准化（RF通常不需要，但保持一致性）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 模拟退火特征选择
    print("开始模拟退火特征选择...")
    sa = SimulatedAnnealing(
        initial_temperature=100,     # 初始温度
        final_temperature=0.01,      # 最终温度
        cooling_rate=0.95,           # 冷却率
        max_iterations=1000,         # 最大迭代数
        cv_folds=5                   # 交叉验证折数
    )
    
    sa.fit(X_train_scaled, y_train)
    
    X_train_sa = sa.transform(X_train_scaled)
    X_test_sa = sa.transform(X_test_scaled)
    print(f"模拟退火选择了 {X_train_sa.shape[1]} 个特征")

    # 6. 使用GridSearchCV进行随机森林模型训练和参数优化（使用SA选择的特征）
    print("开始随机森林训练和参数优化...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train_sa, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {best_model}")

    # 7. 模型评估
    y_pred = best_model.predict(X_test_sa)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- 模型评估结果 ---")
    print(f"准确率: {accuracy:.2f}")
    print(f"精准率: {precision:.2f}")
    print(f"召回率: {recall:.2f}")
    print(f"F1值: {f1:.2f}")
    
    # 显示特征选择信息
    selected_features = sa.get_selected_features()
    print(f"\n选择的特征索引: {selected_features}")
    print(f"特征选择率: {len(selected_features)}/{X.shape[1]} = {len(selected_features)/X.shape[1]*100:.2f}%")
    
    # 显示优化过程信息
    opt_info = sa.get_optimization_info()
    print(f"\n优化过程信息:")
    print(f"初始适应度: {sa.fitness_history[0]:.4f}")
    print(f"最终适应度: {sa.fitness_history[-1]:.4f}")
    print(f"最佳适应度: {sa.best_fitness:.4f}")
    print(f"适应度提升: {sa.best_fitness - sa.fitness_history[0]:.4f}")
    print(f"总体接受率: {opt_info['final_acceptance_rate']:.2f}%")
    print(f"初始温度: {sa.initial_temperature}")
    print(f"最终温度: {sa.temperature_history[-1]:.6f}")

if __name__ == '__main__':
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    process_spectrum_data(data_path) 