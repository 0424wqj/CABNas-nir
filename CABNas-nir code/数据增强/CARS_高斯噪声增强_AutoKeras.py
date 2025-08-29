import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, classification_report
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
import autokeras as ak
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# GPU配置
def configure_gpu():
    """配置GPU使用"""
    # 检测GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"检测到 {len(physical_devices)} 个GPU设备:")
    for device in physical_devices:
        print(f"  - {device}")
    
    if len(physical_devices) > 0:
        try:
            # 设置GPU内存增长
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("✅ GPU配置成功，启用内存动态增长")
            
            # 设置混合精度训练（可选，进一步加速）
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ 启用混合精度训练加速")
            
            return True
        except Exception as e:
            print(f"⚠️ GPU配置失败: {e}")
            return False
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练")
        return False

class CARS:
    """CARS特征选择算法"""
    def __init__(self, n_iterations=50, cv_folds=5):
        self.n_iterations = n_iterations
        self.cv_folds = cv_folds
        self.best_feature_indices_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.best_rmse_ = float('inf')
        
        retained_feature_indices_history = []
        retained_indices = list(range(n_features))

        for i in range(self.n_iterations):
            pls = PLSRegression(n_components=min(10, len(retained_indices), n_samples - 1))
            pls.fit(X[:, retained_indices], y)
            weights = np.abs(pls.coef_).flatten()
            
            ratio = (1 / (i + 1)) ** 0.3
            n_retained = max(2, int(n_features * ratio))
            
            sorted_indices = np.argsort(weights)[::-1]
            retained_indices_local = sorted_indices[:n_retained]
            retained_indices = np.array(retained_indices)[retained_indices_local]
            retained_feature_indices_history.append(retained_indices)

            if len(retained_indices) < 2:
                break

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

class GaussianNoiseAugmentation:
    """高斯噪声数据增强器"""
    
    def __init__(self, random_state=42):
        np.random.seed(random_state)
        self.random_state = random_state
    
    def add_gaussian_noise(self, X, noise_level=0.01):
        """
        添加高斯噪声进行数据增强
        模拟仪器测量噪声
        """
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    def augment_data(self, X, y, augmentation_factor=3, noise_level=0.01):
        """
        对数据进行高斯噪声增强
        """
        print(f"🔊 正在进行高斯噪声数据增强...")
        print(f"   增强倍数: {augmentation_factor}")
        print(f"   噪声强度: {noise_level}")
        
        X_augmented_list = [X]  # 包含原始数据
        y_augmented_list = [y]
        
        for i in range(augmentation_factor):
            X_aug = self.add_gaussian_noise(X, noise_level)
            X_augmented_list.append(X_aug)
            y_augmented_list.append(y)
        
        X_augmented = np.vstack(X_augmented_list)
        y_augmented = np.hstack(y_augmented_list)
        
        print(f"   原始数据量: {len(X)} -> 增强后数据量: {len(X_augmented)}")
        
        return X_augmented, y_augmented

def print_model_architecture(model, model_name="AutoKeras模型"):
    """打印AutoKeras搜索出来的网络结构"""
    print(f"\n" + "="*70)
    print(f"           {model_name} - 搜索出的网络架构详情")
    print("="*70)
    
    try:
        if hasattr(model, 'export_model'):
            best_model = model.export_model()
            print("📋 AutoKeras搜索出的最佳网络架构摘要:")
            print("-" * 70)
            best_model.summary()
            print(f"\n📊 模型参数数量: {best_model.count_params():,}")
            
            # 分析网络架构类型
            layer_types = [layer.__class__.__name__ for layer in best_model.layers]
            has_conv = any('Conv' in layer_type for layer_type in layer_types)
            has_lstm = any('LSTM' in layer_type for layer_type in layer_types)
            has_gru = any('GRU' in layer_type for layer_type in layer_types)
            has_dense = any('Dense' in layer_type for layer_type in layer_types)
            
            architecture_components = []
            if has_conv:
                architecture_components.append("CNN")
            if has_lstm:
                architecture_components.append("LSTM")
            if has_gru:
                architecture_components.append("GRU")
            if has_dense:
                architecture_components.append("Dense")
            
            print(f"🎯 架构组成: {' + '.join(architecture_components)}")
                
        else:
            print("⚠️ 无法获取详细架构信息")
            
    except Exception as e:
        print(f"❌ 打印架构时出错: {str(e)}")
    
    print("="*70)

def process_spectrum_data_with_gaussian_noise_augmentation(file_path):
    """
    CARS特征选择 + 高斯噪声数据增强 + AutoKeras CNN+RNN
    """
    print("="*80)
    print("        CARS + 高斯噪声增强 + AutoKeras CNN+RNN")
    print("="*80)
    
    # 1. 加载数据
    print("步骤1: 正在加载数据...")
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    print(f"原始数据形状: {X.shape}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 2. 标签编码
    print("\n步骤2: 正在进行标签编码...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 3. SG预处理
    print("\n步骤3: 正在进行SG预处理...")
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 4. 数据划分
    print("\n步骤4: 正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y_encoded, test_size=0.3, random_state=45, stratify=y_encoded
    )

    # 5. 数据标准化
    print("\n步骤5: 正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. CARS特征选择
    print("\n步骤6: 正在进行CARS特征选择...")
    cars = CARS(n_iterations=50, cv_folds=5)
    cars.fit(X_train_scaled, y_train)
    
    X_train_cars = cars.transform(X_train_scaled)
    X_test_cars = cars.transform(X_test_scaled)
    print(f"CARS选择了 {X_train_cars.shape[1]} 个特征")
    print(f"特征选择率: {X_train_cars.shape[1]}/{X.shape[1]} = {X_train_cars.shape[1]/X.shape[1]*100:.2f}%")

    # 7. 高斯噪声数据增强
    print("\n步骤7: 正在进行高斯噪声数据增强...")
    augmenter = GaussianNoiseAugmentation(random_state=42)
    X_train_augmented, y_train_augmented = augmenter.augment_data(
        X_train_cars, y_train, 
        augmentation_factor=3,  # 增强3倍
        noise_level=0.01        # 噪声强度
    )

    # 8. 调整数据形状以适应CNN+RNN
    print("\n步骤8: 正在调整数据形状...")
    X_train_reshaped = X_train_augmented.reshape(X_train_augmented.shape[0], 1, X_train_augmented.shape[1])
    X_test_reshaped = X_test_cars.reshape(X_test_cars.shape[0], 1, X_test_cars.shape[1])
    y_train_final = y_train_augmented.astype(np.int32)
    y_test_final = y_test.astype(np.int32)
    
    print(f"训练数据形状: {X_train_reshaped.shape}")
    print(f"测试数据形状: {X_test_reshaped.shape}")

    # 9. 创建AutoKeras模型
    print("\n步骤9: 正在创建AutoKeras CNN+RNN模型...")
    input_node = ak.Input()
    output_node = ak.Normalization()(input_node)
    output_node = ak.ConvBlock(num_blocks=2, num_layers=2, dropout=0.1)(output_node)
    output_node = ak.RNNBlock(layer_type='lstm', num_layers=1, bidirectional=False)(output_node)
    output_node = ak.ClassificationHead()(output_node)
    
    model = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=8
    )
    
    print("正在开始AutoKeras模型搜索和训练...")
    
    # 10. 训练模型
    model.fit(
        X_train_reshaped, 
        y_train_final,
        validation_split=0.2,
        epochs=800,
        verbose=1
    )
    
    print("AutoKeras模型训练完成！")
    
    # 打印网络结构
    print_model_architecture(model, "CARS + 高斯噪声增强 + AutoKeras CNN+RNN")
    
    # 11. 模型评估
    print("\n步骤10: 正在进行模型评估...")
    y_pred = model.predict(X_test_reshaped)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_final = np.argmax(y_pred, axis=1)
    else:
        y_pred_final = y_pred.flatten().astype(np.int32)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test_final, y_pred_final) * 100
    precision = precision_score(y_test_final, y_pred_final, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test_final, y_pred_final, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test_final, y_pred_final, average='weighted', zero_division=0) * 100

    print("\n" + "="*80)
    print("       CARS + 高斯噪声增强 + AutoKeras 模型评估结果")
    print("="*80)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    print("="*80)
    
    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(y_test_final, y_pred_final, target_names=[str(label) for label in le.classes_]))
    
    # 显示增强效果总结
    print(f"\n🎯 高斯噪声增强效果总结:")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"CARS选择特征: {X_train_cars.shape[1]} ({X_train_cars.shape[1]/X.shape[1]*100:.1f}%)")
    print(f"原始训练样本: {len(X_train)}")
    print(f"增强后样本: {len(X_train_augmented)} ({len(X_train_augmented)/len(X_train):.1f}倍)")
    print(f"增强方法: 高斯噪声 (σ=0.01)")
    print(f"最终模型性能: {accuracy:.2f}%")
    
    # 获取最终模型
    exported_model = model.export_model()
    
    return {
        'method': '高斯噪声增强',
        'model': exported_model,
        'scaler': scaler,
        'cars': cars,
        'label_encoder': le,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'original_features': X.shape[1],
        'selected_features': X_train_cars.shape[1],
        'original_samples': len(X_train),
        'augmented_samples': len(X_train_augmented),
        'augmentation_factor': len(X_train_augmented)/len(X_train)
    }

if __name__ == '__main__':
    # 配置GPU
    gpu_available = configure_gpu()
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("🔊 启动CARS + 高斯噪声增强 + AutoKeras CNN+RNN实验...")
    if gpu_available:
        print("🚀 使用GPU加速训练")
    
    try:
        result = process_spectrum_data_with_gaussian_noise_augmentation(data_path)
        
        print(f"\n🎉 高斯噪声增强实验完成！")
        print(f"最终模型性能: {result['accuracy']:.2f}%")
        print(f"数据增强倍数: {result['augmentation_factor']:.1f}×")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}") 