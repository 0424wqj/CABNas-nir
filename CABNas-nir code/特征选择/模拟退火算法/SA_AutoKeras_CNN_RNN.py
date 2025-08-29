import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import savgol_filter
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import autokeras as ak
import tensorflow as tf
import random
import math
import warnings
warnings.filterwarnings('ignore')

class Solution:
    """解类，表示一个特征选择方案"""
    def __init__(self, n_features, total_features):
        self.n_features = n_features
        self.total_features = total_features
        # 随机初始化特征选择方案
        self.selected_indices = np.random.choice(total_features, n_features, replace=False)
        self.fitness = 0.0

    def copy(self):
        """创建解的副本"""
        new_solution = Solution(self.n_features, self.total_features)
        new_solution.selected_indices = self.selected_indices.copy()
        new_solution.fitness = self.fitness
        return new_solution

    def get_neighbor(self):
        """生成邻域解：随机替换1-3个特征"""
        neighbor = self.copy()
        
        # 随机选择要替换的特征数量（1-3个）
        n_changes = random.randint(1, min(3, self.n_features))
        
        # 随机选择要替换的特征位置
        change_positions = np.random.choice(self.n_features, n_changes, replace=False)
        
        # 获取未选择的特征
        available_features = list(set(range(self.total_features)) - set(neighbor.selected_indices))
        
        # 替换特征
        if len(available_features) >= n_changes:
            new_features = np.random.choice(available_features, n_changes, replace=False)
            neighbor.selected_indices[change_positions] = new_features
        
        return neighbor

class SA_FeatureSelection:
    """模拟退火特征选择算法"""
    
    def __init__(self, n_features=50, initial_temperature=50, final_temperature=0.01, 
                 cooling_rate=0.95, max_iterations=300, cv_folds=3):
        self.n_features = n_features
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
        """适应度函数：基于选择的特征进行SVM交叉验证"""
        selected_features = solution.selected_indices
        
        if len(selected_features) < 2:
            return 0.0
        
        X_selected = X[:, selected_features]
        
        # 使用SVM进行快速评估
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        
        try:
            # 交叉验证评估
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, val_idx in kf.split(X_selected):
                X_train_cv, X_val_cv = X_selected[train_idx], X_selected[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                model.fit(X_train_cv, y_train_cv)
                y_pred_cv = model.predict(X_val_cv)
                accuracy = accuracy_score(y_val_cv, y_pred_cv)
                accuracies.append(accuracy)
            
            avg_accuracy = np.mean(accuracies)
            
            # 添加特征数量惩罚项，鼓励选择较少的特征
            feature_ratio = len(selected_features) / X.shape[1]
            feature_penalty = feature_ratio * 0.05
            fitness = avg_accuracy - feature_penalty
            
            return fitness
            
        except Exception as e:
            return 0.0

    def accept_solution(self, current_fitness, new_fitness, temperature):
        """Metropolis接受准则"""
        if new_fitness > current_fitness:
            # 新解更好，直接接受
            return True
        else:
            # 新解较差，按概率接受
            if temperature > 0:
                delta = new_fitness - current_fitness
                probability = math.exp(delta / temperature)
                return random.random() < probability
            else:
                return False

    def cool_down(self, temperature):
        """冷却策略：指数衰减"""
        return temperature * self.cooling_rate

    def fit(self, X, y):
        """训练模拟退火特征选择算法"""
        total_features = X.shape[1]
        
        # 初始化当前解
        current_solution = Solution(self.n_features, total_features)
        current_solution.fitness = self.fitness_function(X, y, current_solution)
        
        # 初始化最佳解
        self.best_solution = current_solution.copy()
        self.best_fitness = current_solution.fitness
        
        # 初始化温度
        temperature = self.initial_temperature
        
        print(f"开始模拟退火特征选择")
        print(f"初始温度: {self.initial_temperature}, 最终温度: {self.final_temperature}")
        print(f"冷却率: {self.cooling_rate}, 最大迭代数: {self.max_iterations}")
        print(f"目标特征数: {self.n_features}, 总特征数: {total_features}")
        
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
            if iteration % 50 == 0:
                acceptance_rate = accepted_count / iteration * 100
                print(f"第 {iteration:3d}/{self.max_iterations} 代: 温度={temperature:.4f}, "
                      f"当前适应度={current_solution.fitness:.4f}, "
                      f"最佳适应度={self.best_fitness:.4f}, "
                      f"接受率={acceptance_rate:.1f}%")
        
        final_acceptance_rate = accepted_count / iteration * 100 if iteration > 0 else 0
        print(f"\n模拟退火优化完成！")
        print(f"总迭代次数: {iteration}")
        print(f"最终温度: {temperature:.6f}")
        print(f"总体接受率: {final_acceptance_rate:.2f}%")
        print(f"最佳适应度: {self.best_fitness:.4f}")
        print(f"选择了 {len(self.best_solution.selected_indices)} 个特征")
        
        return self

    def transform(self, X):
        """使用最佳解的特征选择方案转换数据"""
        if self.best_solution is None:
            raise ValueError("请先调用fit方法训练模型")
        
        return X[:, self.best_solution.selected_indices]

    def get_selected_features(self):
        """获取选择的特征索引"""
        if self.best_solution is None:
            return None
        return self.best_solution.selected_indices

    def get_optimization_info(self):
        """获取优化过程信息"""
        return {
            'temperature_history': self.temperature_history,
            'fitness_history': self.fitness_history,
            'acceptance_history': self.acceptance_history,
            'final_acceptance_rate': sum(self.acceptance_history) / len(self.acceptance_history) * 100 if self.acceptance_history else 0
        }

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
                if hasattr(layer, 'strides') and layer.strides:
                    print(f"       步长: {layer.strides}")
                if hasattr(layer, 'padding') and layer.padding:
                    print(f"       填充方式: {layer.padding}")
                if hasattr(layer, 'activation') and layer.activation:
                    activation_name = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
                    print(f"       激活函数: {activation_name}")
                if hasattr(layer, 'dropout') and hasattr(layer, 'rate'):
                    print(f"       Dropout率: {layer.rate}")
                if hasattr(layer, 'units') and hasattr(layer, 'return_sequences'):
                    print(f"       返回序列: {layer.return_sequences}")
                if hasattr(layer, 'go_backwards'):
                    print(f"       双向: {hasattr(layer, 'backward_layer')}")
                
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
            has_gru = any('GRU' in layer_type for layer_type in layer_types)
            has_rnn = any('RNN' in layer_type or 'LSTM' in layer_type or 'GRU' in layer_type for layer_type in layer_types)
            has_dense = any('Dense' in layer_type for layer_type in layer_types)
            has_dropout = any('Dropout' in layer_type for layer_type in layer_types)
            has_batch_norm = any('BatchNorm' in layer_type for layer_type in layer_types)
            
            architecture_components = []
            if has_conv:
                architecture_components.append("CNN（卷积神经网络）")
            if has_lstm:
                architecture_components.append("LSTM（长短期记忆网络）")
            if has_gru:
                architecture_components.append("GRU（门控循环单元）")
            elif has_rnn and not has_lstm:
                architecture_components.append("RNN（循环神经网络）")
            if has_dense:
                architecture_components.append("全连接层")
            
            print(f"架构组成: {' + '.join(architecture_components)}")
            
            regularization_techniques = []
            if has_dropout:
                regularization_techniques.append("Dropout")
            if has_batch_norm:
                regularization_techniques.append("批归一化")
            
            if regularization_techniques:
                print(f"正则化技术: {', '.join(regularization_techniques)}")
            
            # 统计各层类型数量
            layer_counts = {}
            for layer_type in layer_types:
                layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            
            print(f"\n各层类型统计:")
            for layer_type, count in layer_counts.items():
                print(f"  {layer_type}: {count}层")
                
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

    # 6. 模拟退火特征选择
    print("正在进行模拟退火特征选择...")
    sa = SA_FeatureSelection(
        n_features=min(80, X.shape[1]//3),  # 动态调整特征数
        initial_temperature=50,       # 初始温度
        final_temperature=0.01,       # 最终温度
        cooling_rate=0.95,           # 冷却率
        max_iterations=200,          # 最大迭代数
        cv_folds=3                   # 交叉验证折数
    )
    
    sa.fit(X_train_scaled, y_train)
    
    X_train_sa = sa.transform(X_train_scaled)
    X_test_sa = sa.transform(X_test_scaled)
    print(f"模拟退火选择了 {X_train_sa.shape[1]} 个特征")
    print(f"特征选择率: {X_train_sa.shape[1]}/{X.shape[1]} = {X_train_sa.shape[1]/X.shape[1]*100:.2f}%")

    # 7. 调整数据形状以适应CNN+RNN (samples, timesteps, features)
    print("正在调整数据形状...")
    # 将数据reshape为3D: (samples, 1, features) 适合1D卷积处理
    X_train_reshaped = X_train_sa.reshape(X_train_sa.shape[0], 1, X_train_sa.shape[1])
    X_test_reshaped = X_test_sa.reshape(X_test_sa.shape[0], 1, X_test_sa.shape[1])
    
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
        max_trials=10
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
    print_model_architecture(autokeras_model, "SA + AutoKeras CNN+RNN 搜索结果")
    
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

    print("\n" + "="*60)
    print("     SA + AutoKeras CNN+RNN 模型评估结果")
    print("="*60)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    
    # 显示分类报告
    from sklearn.metrics import classification_report
    print("\n分类报告:")
    print(classification_report(y_test_final, y_pred_final, target_names=[str(label) for label in le.classes_]))
    print("="*60)
    
    # 显示特征选择信息
    selected_features = sa.get_selected_features()
    print(f"\n模拟退火选择的特征索引: {selected_features}")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"特征选择率: {len(selected_features)/X.shape[1]*100:.2f}%")
    print(f"模拟退火最佳适应度: {sa.best_fitness:.4f}")
    
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
    
    # 显示模型架构信息
    print(f"\n模型架构信息:")
    print(f"输入形状: {X_train_reshaped.shape[1:]}")
    print(f"类别数量: {len(le.classes_)}")
    print(f"架构: 简化CNN(特征提取) + LSTM(序列建模) + 分类头")
    print(f"特征选择方法: 模拟退火算法 (SA)")
    
    # 获取最终模型
    print("\n正在导出最佳模型...")
    final_model = autokeras_model.export_model()
    print("模型导出成功！")
    
    return final_model, scaler, sa, le

if __name__ == '__main__':
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("正在启动基于模拟退火特征选择的AutoKeras CNN+RNN分类流程...")
    print("="*70)
    
    result = process_spectrum_data(data_path)
    
    if result[0] is not None:
        print("\n流程执行成功！")
        print("已生成：")
        print("1. 训练好的AutoKeras CNN+RNN模型")
        print("2. 数据标准化器")
        print("3. 模拟退火特征选择器")
        print("4. 标签编码器")
    else:
        print("\n流程执行遇到问题，请检查环境配置。") 