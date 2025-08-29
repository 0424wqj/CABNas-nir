import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter
import autokeras as ak
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

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
        """适应度函数：基于选择的特征进行简单的交叉验证"""
        selected_features = np.where(individual.chromosome == 1)[0]
        
        if len(selected_features) < 2:
            return 0.0
        
        X_selected = X[:, selected_features]
        
        # 使用简化的交叉验证评估特征子集（减少计算量）
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        accuracies = []
        
        for train_idx, val_idx in kf.split(X_selected):
            X_train_cv, X_val_cv = X_selected[train_idx], X_selected[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # 使用简单的逻辑回归评估（比SVM更快）
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(random_state=42, max_iter=100)
            lr.fit(X_train_cv, y_train_cv)
            y_pred_cv = lr.predict(X_val_cv)
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

def print_model_architecture(model, model_name="AutoKeras模型"):
    """打印AutoKeras搜索出来的网络结构"""
    print(f"\n" + "="*70)
    print(f"           {model_name} - 搜索出的网络架构详情")
    print("="*70)
    
    try:
        # 获取最佳模型
        if hasattr(model, 'export_model'):
            best_model = model.export_model()
            
            print("📋 AutoKeras搜索出的最佳网络架构摘要:")
            print("-" * 70)
            best_model.summary()
            
            print(f"\n📊 模型详细信息:")
            print(f"总参数数量: {best_model.count_params():,}")
            print(f"可训练参数: {sum([tf.keras.backend.count_params(w) for w in best_model.trainable_weights]):,}")
            print(f"不可训练参数: {sum([tf.keras.backend.count_params(w) for w in best_model.non_trainable_weights]):,}")
            
            # 打印每一层的详细信息
            print(f"\n🏗️ 层结构详细信息:")
            print("-" * 70)
            for i, layer in enumerate(best_model.layers):
                print(f"第{i+1:2d}层: {layer.__class__.__name__}")
                print(f"       名称: {layer.name}")
                print(f"       输出形状: {layer.output_shape}")
                
                # 根据层类型显示特定信息
                if hasattr(layer, 'units') and layer.units:
                    print(f"       单元数: {layer.units}")
                if hasattr(layer, 'filters') and layer.filters:
                    print(f"       滤波器数: {layer.filters}")
                if hasattr(layer, 'kernel_size') and layer.kernel_size:
                    print(f"       卷积核大小: {layer.kernel_size}")
                if hasattr(layer, 'activation') and layer.activation:
                    activation_name = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
                    print(f"       激活函数: {activation_name}")
                
                # 显示参数数量
                layer_params = layer.count_params()
                if layer_params > 0:
                    print(f"       参数数量: {layer_params:,}")
                
                print()
            
            # 分析网络架构类型
            print("🎯 网络架构分析:")
            print("-" * 70)
            layer_types = [layer.__class__.__name__ for layer in best_model.layers]
            
            has_conv = any('Conv' in layer_type for layer_type in layer_types)
            has_lstm = any('LSTM' in layer_type for layer_type in layer_types)
            has_rnn = any('RNN' in layer_type or 'LSTM' in layer_type or 'GRU' in layer_type for layer_type in layer_types)
            has_dense = any('Dense' in layer_type for layer_type in layer_types)
            
            architecture_components = []
            if has_conv:
                architecture_components.append("CNN（卷积神经网络）")
            if has_lstm:
                architecture_components.append("LSTM（长短期记忆网络）")
            elif has_rnn:
                architecture_components.append("RNN（循环神经网络）")
            if has_dense:
                architecture_components.append("全连接层")
            
            print(f"架构组成: {' + '.join(architecture_components)}")
                
        else:
            print("⚠️ 无法获取详细架构信息 - 模型可能尚未训练完成")
            
    except Exception as e:
        print(f"❌ 打印架构时出错: {str(e)}")
        print("这可能是由于模型尚未完全训练完成")
    
    print("="*70)

def process_spectrum_data(file_path):
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # 2. 标签编码（AutoKeras需要从0开始的整数标签）
    print("正在进行标签编码...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
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

    # 6. 遗传算法特征选择
    print("正在进行遗传算法特征选择...")
    ga = GeneticAlgorithm(
        population_size=30,     # 减少种群大小以加快速度
        generations=15,         # 减少进化代数
        crossover_rate=0.8,     # 交叉概率
        mutation_rate=0.05,     # 变异概率
        elite_size=2,           # 精英个体数量
        cv_folds=3              # 减少交叉验证折数以加快速度
    )
    
    ga.fit(X_train_scaled, y_train)
    
    X_train_ga = ga.transform(X_train_scaled)
    X_test_ga = ga.transform(X_test_scaled)
    print(f"遗传算法选择了 {X_train_ga.shape[1]} 个特征")
    print(f"特征选择率: {X_train_ga.shape[1]}/{X.shape[1]} = {X_train_ga.shape[1]/X.shape[1]*100:.2f}%")

    # 7. 调整数据形状以适应CNN+RNN (samples, timesteps, features)
    print("正在调整数据形状...")
    # 将数据reshape为3D: (samples, 1, features) 适合1D卷积处理
    X_train_reshaped = X_train_ga.reshape(X_train_ga.shape[0], 1, X_train_ga.shape[1])
    X_test_reshaped = X_test_ga.reshape(X_test_ga.shape[0], 1, X_test_ga.shape[1])
    
    # 确保标签为正确的形状和类型
    y_train_final = y_train.astype(np.int32)
    y_test_final = y_test.astype(np.int32)
    
    print(f"训练数据形状: {X_train_reshaped.shape}")
    print(f"测试数据形状: {X_test_reshaped.shape}")
    print(f"训练标签形状: {y_train_final.shape}")
    print(f"测试标签形状: {y_test_final.shape}")

    # 8. 创建简化的AutoKeras CNN+RNN模型
    print("正在创建简化的AutoKeras CNN+RNN模型...")
    
    # 简化的架构
    input_node = ak.Input()
    output_node = ak.Normalization()(input_node)
    output_node = ak.ConvBlock(num_blocks=1, num_layers=2, dropout=0.1)(output_node)
    output_node = ak.RNNBlock(layer_type='lstm', num_layers=1, bidirectional=False)(output_node)
    output_node = ak.ClassificationHead()(output_node)
    
    autokeras_model = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=5
    )
    
    print("正在开始AutoKeras模型搜索和训练...")
    print("这可能需要较长时间，请耐心等待...")
    
    # 训练模型
    autokeras_model.fit(
        X_train_reshaped, 
        y_train_final,
        validation_split=0.2,
        epochs=800,
        verbose=1
    )
    
    print("AutoKeras模型训练完成！")
    
    # 打印搜索出来的网络结构
    print_model_architecture(autokeras_model, "GA+AutoKeras CNN+RNN 搜索结果")
    
    # 9. 模型评估
    print("正在进行模型评估...")
    y_pred = autokeras_model.predict(X_test_reshaped)
    
    # 处理预测结果
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_final = np.argmax(y_pred, axis=1)
    else:
        y_pred_final = y_pred.flatten().astype(np.int32)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test_final, y_pred_final) * 100
    precision = precision_score(y_test_final, y_pred_final, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test_final, y_pred_final, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test_final, y_pred_final, average='weighted', zero_division=0) * 100

    print("\n" + "="*50)
    print("        GA+AutoKeras CNN+RNN 模型评估结果")
    print("="*50)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    print("="*50)
    
    # 显示特征选择信息
    selected_features = ga.get_selected_features()
    print(f"\nGA选择的特征索引: {selected_features}")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"特征选择率: {len(selected_features)/X.shape[1]*100:.2f}%")
    
    # 显示模型架构信息
    print(f"\n模型架构信息:")
    print(f"输入形状: {X_train_reshaped.shape[1:]}")
    print(f"类别数量: {len(le.classes_)}")
    print(f"架构: GA特征选择 + 简化CNN(特征提取) + LSTM(序列建模) + 分类头")
    
    # 显示进化过程
    print(f"\n遗传算法进化过程信息:")
    print(f"初始适应度: {ga.fitness_history[0]:.4f}")
    print(f"最终适应度: {ga.fitness_history[-1]:.4f}")
    print(f"适应度提升: {ga.fitness_history[-1] - ga.fitness_history[0]:.4f}")
    
    # 获取最终模型
    print("\n正在导出最佳模型...")
    final_model = autokeras_model.export_model()
    print("模型导出成功！")
    
    return final_model, scaler, ga, le

if __name__ == '__main__':
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("正在启动基于GA特征选择的AutoKeras CNN+RNN分类流程...")
    print("="*60)
    
    result = process_spectrum_data(data_path)
    
    if result[0] is not None:
        print("\n流程执行成功！")
        print("已生成：")
        print("1. 训练好的GA+AutoKeras CNN+RNN模型")
        print("2. 数据标准化器")
        print("3. GA特征选择器")
        print("4. 标签编码器")
    else:
        print("\n流程执行遇到问题，请检查环境配置。") 