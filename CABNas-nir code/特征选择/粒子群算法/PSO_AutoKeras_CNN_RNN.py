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
import warnings
warnings.filterwarnings('ignore')

class PSO_FeatureSelection:
    """粒子群特征选择算法"""
    
    def __init__(self, n_features=50, n_particles=20, max_iter=50, w=0.9, c1=2, c2=2, cv_folds=3):
        self.n_features = n_features
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.cv_folds = cv_folds
        self.selected_indices = None
        self.best_fitness = 0.0
        
    def objective_function(self, X, y, indices):
        """目标函数：使用交叉验证评估特征子集"""
        if len(indices) < 2:
            return 0.0
        
        X_selected = X[:, indices]
        
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
            feature_ratio = len(indices) / X.shape[1]
            feature_penalty = feature_ratio * 0.05
            fitness = avg_accuracy - feature_penalty
            
            return fitness
            
        except Exception as e:
            return 0.0
    
    def fit(self, X, y):
        """粒子群特征选择"""
        print(f"开始粒子群特征选择")
        print(f"粒子数量: {self.n_particles}, 最大迭代数: {self.max_iter}")
        print(f"目标特征数: {self.n_features}, 总特征数: {X.shape[1]}")
        
        n_total_features = X.shape[1]
        
        # 初始化粒子群
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_scores = []
        
        print("正在初始化粒子群...")
        for i in range(self.n_particles):
            # 随机初始化粒子位置（特征索引）
            position = np.random.choice(n_total_features, self.n_features, replace=False)
            particles.append(position)
            
            # 初始化速度
            velocity = np.random.randint(-5, 5, size=self.n_features)
            velocities.append(velocity)
            
            # 评估初始位置
            score = self.objective_function(X, y, position)
            personal_best_positions.append(position.copy())
            personal_best_scores.append(score)
            
            if (i + 1) % 5 == 0:
                print(f"  已初始化 {i + 1}/{self.n_particles} 个粒子")
        
        # 找到全局最优
        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        print(f"初始全局最优适应度: {global_best_score:.4f}")
        
        # PSO迭代
        for iteration in range(self.max_iter):
            print(f"第 {iteration + 1}/{self.max_iter} 代")
            
            for i in range(self.n_particles):
                # 更新速度和位置
                r1, r2 = np.random.random(2)
                
                # 计算个体和社会学习项
                personal_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                global_component = self.c2 * r2 * (global_best_position - particles[i])
                
                # 更新速度
                velocities[i] = (self.w * velocities[i] + 
                               personal_component.astype(int) + 
                               global_component.astype(int))
                
                # 更新位置
                new_position = particles[i] + velocities[i]
                
                # 确保位置在有效范围内
                new_position = np.clip(new_position, 0, n_total_features - 1)
                new_position = np.unique(new_position.astype(int))
                
                # 调整特征数量
                if len(new_position) < self.n_features:
                    # 特征数不够，随机补充
                    available_features = list(set(range(n_total_features)) - set(new_position))
                    if len(available_features) > 0:
                        additional_needed = min(self.n_features - len(new_position), len(available_features))
                        additional_features = np.random.choice(
                            available_features, 
                            additional_needed, 
                            replace=False
                        )
                        new_position = np.concatenate([new_position, additional_features])
                elif len(new_position) > self.n_features:
                    # 特征数太多，随机选择
                    new_position = np.random.choice(new_position, self.n_features, replace=False)
                
                particles[i] = new_position
                
                # 评估新位置
                score = self.objective_function(X, y, new_position)
                
                # 更新个体最优
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = new_position.copy()
                    
                    # 检查是否需要更新全局最优
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = new_position.copy()
                        print(f"  找到更好的全局最优! 适应度: {global_best_score:.4f}")
            
            # 动态调整惯性权重
            self.w = self.w * 0.95
            
            print(f"  当前最佳适应度: {global_best_score:.4f}")
            print(f"  当前选择特征数: {len(global_best_position)}")
        
        self.selected_indices = global_best_position
        self.best_fitness = global_best_score
        
        print(f"粒子群特征选择完成！")
        print(f"最佳适应度: {self.best_fitness:.4f}")
        print(f"选择了 {len(self.selected_indices)} 个特征")
        
        return self
    
    def transform(self, X):
        """转换数据"""
        if self.selected_indices is None:
            raise ValueError("请先调用fit方法进行特征选择")
        return X[:, self.selected_indices]
    
    def get_selected_features(self):
        """获取选择的特征索引"""
        return self.selected_indices

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

    # 6. 粒子群特征选择
    print("正在进行粒子群特征选择...")
    pso = PSO_FeatureSelection(
        n_features=min(100, X.shape[1]//4),  # 动态调整特征数
        n_particles=20,        # 粒子数量
        max_iter=50,          # 最大迭代数
        w=0.9,                # 惯性权重
        c1=2,                 # 个体学习因子
        c2=2,                 # 社会学习因子
        cv_folds=3            # 交叉验证折数
    )
    
    pso.fit(X_train_scaled, y_train)
    
    X_train_pso = pso.transform(X_train_scaled)
    X_test_pso = pso.transform(X_test_scaled)
    print(f"粒子群选择了 {X_train_pso.shape[1]} 个特征")
    print(f"特征选择率: {X_train_pso.shape[1]}/{X.shape[1]} = {X_train_pso.shape[1]/X.shape[1]*100:.2f}%")

    # 7. 调整数据形状以适应CNN+RNN (samples, timesteps, features)
    print("正在调整数据形状...")
    # 将数据reshape为3D: (samples, 1, features) 适合1D卷积处理
    X_train_reshaped = X_train_pso.reshape(X_train_pso.shape[0], 1, X_train_pso.shape[1])
    X_test_reshaped = X_test_pso.reshape(X_test_pso.shape[0], 1, X_test_pso.shape[1])
    
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
    print_model_architecture(autokeras_model, "PSO + AutoKeras CNN+RNN 搜索结果")
    
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
    print("     PSO + AutoKeras CNN+RNN 模型评估结果")
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
    selected_features = pso.get_selected_features()
    print(f"\n粒子群选择的特征索引: {selected_features}")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"特征选择率: {len(selected_features)/X.shape[1]*100:.2f}%")
    print(f"粒子群最佳适应度: {pso.best_fitness:.4f}")
    
    # 显示模型架构信息
    print(f"\n模型架构信息:")
    print(f"输入形状: {X_train_reshaped.shape[1:]}")
    print(f"类别数量: {len(le.classes_)}")
    print(f"架构: 简化CNN(特征提取) + LSTM(序列建模) + 分类头")
    print(f"特征选择方法: 粒子群优化 (PSO)")
    
    # 获取最终模型
    print("\n正在导出最佳模型...")
    final_model = autokeras_model.export_model()
    print("模型导出成功！")
    
    return final_model, scaler, pso, le

if __name__ == '__main__':
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("正在启动基于粒子群特征选择的AutoKeras CNN+RNN分类流程...")
    print("="*70)
    
    result = process_spectrum_data(data_path)
    
    if result[0] is not None:
        print("\n流程执行成功！")
        print("已生成：")
        print("1. 训练好的AutoKeras CNN+RNN模型")
        print("2. 数据标准化器")
        print("3. 粒子群特征选择器")
        print("4. 标签编码器")
    else:
        print("\n流程执行遇到问题，请检查环境配置。") 