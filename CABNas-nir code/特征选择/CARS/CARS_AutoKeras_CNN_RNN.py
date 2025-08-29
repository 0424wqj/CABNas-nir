import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
import autokeras as ak
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

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
        print("正在评估特征子集...")
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
    # 1. 加载数据，与CARS_SVM.py保持一致
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
        X_sg, y_encoded, test_size=0.3, random_state=43, stratify=y_encoded
    )

    # 5. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. CARS特征选择
    print("正在进行CARS特征选择...")
    cars = CARS(n_iterations=50, cv_folds=5)
    cars.fit(X_train_scaled, y_train)
    
    X_train_cars = cars.transform(X_train_scaled)
    X_test_cars = cars.transform(X_test_scaled)
    print(f"CARS选择了 {X_train_cars.shape[1]} 个特征")
    print(f"特征选择率: {X_train_cars.shape[1]}/{X.shape[1]} = {X_train_cars.shape[1]/X.shape[1]*100:.2f}%")

    # 7. 调整数据形状以适应CNN+RNN (samples, timesteps, features)
    print("正在调整数据形状...")
    # 将数据reshape为3D: (samples, 1, features) 适合1D卷积处理
    X_train_reshaped = X_train_cars.reshape(X_train_cars.shape[0], 1, X_train_cars.shape[1])
    X_test_reshaped = X_test_cars.reshape(X_test_cars.shape[0], 1, X_test_cars.shape[1])
    
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
    print_model_architecture(autokeras_model, "AutoKeras CNN+RNN 搜索结果")
    
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
    print("           AutoKeras CNN+RNN 模型评估结果")
    print("="*50)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    print("="*50)
    
    # 显示特征选择信息
    selected_features = cars.best_feature_indices_
    print(f"\nCARS选择的特征索引: {selected_features}")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"特征选择率: {len(selected_features)/X.shape[1]*100:.2f}%")
    
    # 显示模型架构信息
    print(f"\n模型架构信息:")
    print(f"输入形状: {X_train_reshaped.shape[1:]}")
    print(f"类别数量: {len(le.classes_)}")
    print(f"架构: 简化CNN(特征提取) + LSTM(序列建模) + 分类头")
    
    # 获取最终模型
    print("\n正在导出最佳模型...")
    final_model = autokeras_model.export_model()
    print("模型导出成功！")
    
    return final_model, scaler, cars, le

if __name__ == '__main__':
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("正在启动基于CARS特征选择的AutoKeras CNN+RNN分类流程...")
    print("="*60)
    
    result = process_spectrum_data(data_path)
    
    if result[0] is not None:
        print("\n流程执行成功！")
        print("已生成：")
        print("1. 训练好的AutoKeras CNN+RNN模型")
        print("2. 数据标准化器")
        print("3. CARS特征选择器")
        print("4. 标签编码器")
    else:
        print("\n流程执行遇到问题，请检查环境配置。") 