import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter
import xgboost as xgb
import random

class Individual:
    """个体类，表示一个特征选择方案"""
    def __init__(self, n_features):
        # 随机初始化个体的染色体（二进制向量，表示特征选择）
        self.chromosome = np.random.randint(0, 2, n_features)
        # 确保至少选择2个特征
        if np.sum(self.chromosome) < 2:
            indices = np.random.choice(n_features, 2, replace=False)
            self.chromosome = np.zeros(n_features)
            self.chromosome[indices] = 1
        
        self.fitness = 0.0
        self.n_features = n_features

    def mutate(self, mutation_rate=0.05):
        """变异操作"""
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                self.chromosome[i] = 1 - self.chromosome[i]
        
        # 确保至少选择2个特征
        if np.sum(self.chromosome) < 2:
            indices = np.random.choice(self.n_features, 2, replace=False)
            self.chromosome[indices] = 1

    def crossover(self, other):
        """交叉操作"""
        # 单点交叉
        crossover_point = random.randint(1, len(self.chromosome) - 1)
        
        child1 = Individual(self.n_features)
        child2 = Individual(self.n_features)
        
        child1.chromosome = np.concatenate([
            self.chromosome[:crossover_point],
            other.chromosome[crossover_point:]
        ])
        
        child2.chromosome = np.concatenate([
            other.chromosome[:crossover_point],
            self.chromosome[crossover_point:]
        ])
        
        # 确保至少选择2个特征
        for child in [child1, child2]:
            if np.sum(child.chromosome) < 2:
                indices = np.random.choice(child.n_features, 2, replace=False)
                child.chromosome = np.zeros(child.n_features)
                child.chromosome[indices] = 1
        
        return child1, child2

class GeneticAlgorithm:
    """遗传算法特征选择"""
    def __init__(self, population_size=50, generations=30, crossover_rate=0.8, 
                 mutation_rate=0.05, elite_size=2, cv_folds=5):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.cv_folds = cv_folds
        self.best_individual = None
        self.best_fitness = 0.0
        self.fitness_history = []

    def fitness_function(self, X, y, individual):
        """适应度函数：基于选择的特征进行XGBoost交叉验证"""
        selected_features = np.where(individual.chromosome == 1)[0]
        
        if len(selected_features) < 2:
            return 0.0
        
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
        feature_penalty = len(selected_features) / X.shape[1] * 0.01
        fitness = avg_accuracy - feature_penalty
        
        return fitness

    def initialize_population(self, n_features):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = Individual(n_features)
            population.append(individual)
        return population

    def evaluate_population(self, X, y, population):
        """评估种群中所有个体的适应度"""
        for individual in population:
            individual.fitness = self.fitness_function(X, y, individual)

    def selection(self, population):
        """锦标赛选择"""
        tournament_size = 3
        selected = []
        
        for _ in range(self.population_size):
            # 随机选择tournament_size个个体进行比赛
            tournament = random.sample(population, tournament_size)
            # 选择适应度最高的个体
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected

    def crossover_and_mutation(self, selected_population):
        """交叉和变异操作"""
        new_population = []
        
        # 保留精英个体
        sorted_population = sorted(selected_population, key=lambda x: x.fitness, reverse=True)
        for i in range(self.elite_size):
            new_population.append(sorted_population[i])
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1, parent2
            
            # 变异
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            
            new_population.extend([child1, child2])
        
        # 确保种群大小正确
        return new_population[:self.population_size]

    def fit(self, X, y):
        """训练遗传算法特征选择"""
        n_features = X.shape[1]
        
        # 初始化种群
        population = self.initialize_population(n_features)
        
        print(f"开始遗传算法特征选择")
        print(f"种群大小: {self.population_size}, 进化代数: {self.generations}")
        print(f"交叉率: {self.crossover_rate}, 变异率: {self.mutation_rate}")
        
        for generation in range(self.generations):
            print(f"第 {generation + 1}/{self.generations} 代")
            
            # 评估种群
            self.evaluate_population(X, y, population)
            
            # 记录当前代最佳个体
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_individual = current_best
            
            self.fitness_history.append(current_best.fitness)
            
            selected_features_count = np.sum(current_best.chromosome)
            print(f"  当前最佳适应度: {current_best.fitness:.4f}")
            print(f"  选择特征数: {selected_features_count}")
            
            # 选择
            selected_population = self.selection(population)
            
            # 交叉和变异
            population = self.crossover_and_mutation(selected_population)
        
        print(f"遗传算法优化完成！")
        print(f"最佳适应度: {self.best_fitness:.4f}")
        print(f"最终选择 {np.sum(self.best_individual.chromosome)} 个特征")
        
        return self

    def transform(self, X):
        """使用最佳个体的特征选择方案转换数据"""
        if self.best_individual is None:
            raise ValueError("请先调用fit方法训练模型")
        
        selected_features = np.where(self.best_individual.chromosome == 1)[0]
        return X[:, selected_features]

    def get_selected_features(self):
        """获取选择的特征索引"""
        if self.best_individual is None:
            return None
        return np.where(self.best_individual.chromosome == 1)[0]

def process_spectrum_data(file_path):
    # 1. 加载数据，与原CARS版本保持一致
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

    # 6. 遗传算法特征选择
    print("开始遗传算法特征选择...")
    ga = GeneticAlgorithm(
        population_size=50,     # 种群大小
        generations=10,         # 进化代数
        crossover_rate=0.8,     # 交叉概率
        mutation_rate=0.05,     # 变异概率
        elite_size=2,           # 精英个体数量
        cv_folds=5              # 交叉验证折数
    )
    
    ga.fit(X_train_scaled, y_train)
    
    X_train_ga = ga.transform(X_train_scaled)
    X_test_ga = ga.transform(X_test_scaled)
    print(f"遗传算法选择了 {X_train_ga.shape[1]} 个特征")

    # 7. 使用GridSearchCV进行XGBoost模型训练和参数优化（使用GA选择的特征）
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
    grid_search.fit(X_train_ga, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {best_model}")

    # 8. 模型评估
    y_pred = best_model.predict(X_test_ga)

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
    selected_features = ga.get_selected_features()
    print(f"\n选择的特征索引: {selected_features}")
    print(f"特征选择率: {len(selected_features)}/{X.shape[1]} = {len(selected_features)/X.shape[1]*100:.2f}%")
    
    # 显示进化过程
    print(f"\n进化过程信息:")
    print(f"初始适应度: {ga.fitness_history[0]:.4f}")
    print(f"最终适应度: {ga.fitness_history[-1]:.4f}")
    print(f"适应度提升: {ga.fitness_history[-1] - ga.fitness_history[0]:.4f}")
    
    # 显示标签映射信息
    print(f"\n标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

if __name__ == '__main__':
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    process_spectrum_data(data_path) 