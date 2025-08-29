import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pywt
import warnings
warnings.filterwarnings('ignore')

# GPU配置
def configure_gpu():
    """配置GPU使用"""
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"检测到 {len(physical_devices)} 个GPU设备:")
    for device in physical_devices:
        print(f"  - {device}")
    
    if len(physical_devices) > 0:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("✅ GPU配置成功，启用内存增长")
            return True
        except Exception as e:
            print(f"⚠️ GPU配置失败: {e}")
            return False
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练")
        return False

class WaveletLayer(tf.keras.layers.Layer):
    """小波变换层"""
    def __init__(self, wavelet='db4', levels=3, **kwargs):
        super(WaveletLayer, self).__init__(**kwargs)
        self.wavelet = wavelet
        self.levels = levels
        
    def build(self, input_shape):
        super(WaveletLayer, self).build(input_shape)
        
    def call(self, inputs):
        """执行小波变换"""
        def wavelet_transform(x):
            # 将张量转换为numpy数组进行小波变换
            batch_size = tf.shape(x)[0]
            
            def single_transform(signal):
                # 对单个信号进行小波变换
                signal_np = signal.numpy()
                coeffs = pywt.wavedec(signal_np, self.wavelet, level=self.levels)
                
                # 提取近似系数和详细系数
                features = []
                for i, coeff in enumerate(coeffs):
                    if i == 0:  # 近似系数
                        features.extend(coeff.tolist())
                    else:  # 详细系数
                        # 计算统计特征
                        features.extend([
                            np.mean(coeff),
                            np.std(coeff),
                            np.var(coeff),
                            np.max(coeff),
                            np.min(coeff)
                        ])
                
                return np.array(features, dtype=np.float32)
            
            # 使用tf.py_function来应用小波变换
            transformed = tf.py_function(
                func=lambda x: tf.stack([single_transform(x[i]) for i in range(tf.shape(x)[0])]),
                inp=[x],
                Tout=tf.float32
            )
            
            return transformed
        
        return wavelet_transform(inputs)
    
    def compute_output_shape(self, input_shape):
        # 估算输出形状（这是一个近似值）
        estimated_length = input_shape[-1] // (2 ** self.levels) + self.levels * 5
        return (input_shape[0], estimated_length)

def wavelet_features_extraction(data, wavelet='db4', levels=3):
    """使用PyWavelets进行小波特征提取"""
    features_list = []
    
    for i, signal in enumerate(data):
        try:
            # 执行小波分解
            coeffs = pywt.wavedec(signal, wavelet, level=levels)
            
            # 提取特征
            features = []
            
            # 近似系数 (低频成分)
            approx = coeffs[0]
            features.extend([
                np.mean(approx),
                np.std(approx),
                np.var(approx),
                np.max(approx),
                np.min(approx),
                np.median(approx)
            ])
            
            # 详细系数 (高频成分)
            for j, detail in enumerate(coeffs[1:], 1):
                features.extend([
                    np.mean(detail),
                    np.std(detail),
                    np.var(detail),
                    np.max(detail),
                    np.min(detail),
                    np.median(detail),
                    np.sum(np.abs(detail)),  # 能量
                    np.sqrt(np.mean(detail**2))  # RMS
                ])
            
            # 添加小波包分解的额外特征
            features.extend([
                np.sum([np.sum(c**2) for c in coeffs]),  # 总能量
                len([c for c in coeffs[0] if abs(c) > np.std(coeffs[0])]),  # 显著系数数量
            ])
            
            features_list.append(features)
            
        except Exception as e:
            print(f"处理第{i}个信号时出错: {e}")
            # 如果出错，使用零填充
            features_list.append([0] * 50)  # 假设特征长度为50
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(data)} 个信号")
    
    return np.array(features_list)

def create_wnn_model(input_shape, wavelet_features_shape, num_classes):
    """创建小波神经网络模型"""
    # 原始信号输入
    signal_input = Input(shape=input_shape, name='signal_input')
    
    # 小波特征输入
    wavelet_input = Input(shape=wavelet_features_shape, name='wavelet_input')
    
    # 原始信号处理分支
    signal_branch = Dense(256, activation='relu')(signal_input)
    signal_branch = Dropout(0.3)(signal_branch)
    signal_branch = Dense(128, activation='relu')(signal_branch)
    signal_branch = Dropout(0.2)(signal_branch)
    
    # 小波特征处理分支
    wavelet_branch = Dense(128, activation='relu')(wavelet_input)
    wavelet_branch = Dropout(0.3)(wavelet_branch)
    wavelet_branch = Dense(64, activation='relu')(wavelet_branch)
    wavelet_branch = Dropout(0.2)(wavelet_branch)
    
    # 特征融合
    merged = Concatenate()([signal_branch, wavelet_branch])
    
    # 分类器
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[signal_input, wavelet_input], outputs=outputs)
    return model

def process_spectrum_data(file_path):
    """
    不使用特征选择的小波神经网络分类
    """
    print("="*60)
    print("      无特征选择 - 小波神经网络(WNN)分类")
    print("="*60)
    
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    print(f"原始数据形状: {X.shape}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 2. 标签编码
    print("正在进行标签编码...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"类别数量: {num_classes}")
    print(f"标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 3. SG预处理
    print("正在进行SG预处理...")
    X_sg = savgol_filter(X, window_length=5, polyorder=2, axis=1)

    # 4. 数据划分
    print("正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sg, y_encoded, test_size=0.3, random_state=45, stratify=y_encoded
    )

    # 5. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"使用全部特征数量: {X_train_scaled.shape[1]}")
    print(f"训练集形状: {X_train_scaled.shape}")
    print(f"测试集形状: {X_test_scaled.shape}")

    # 6. 小波特征提取
    print("正在进行小波特征提取...")
    print("正在提取训练集小波特征...")
    X_train_wavelet = wavelet_features_extraction(X_train_scaled, wavelet='db4', levels=3)
    print("正在提取测试集小波特征...")
    X_test_wavelet = wavelet_features_extraction(X_test_scaled, wavelet='db4', levels=3)
    
    print(f"小波特征形状: {X_train_wavelet.shape}")
    
    # 标准化小波特征
    wavelet_scaler = StandardScaler()
    X_train_wavelet_scaled = wavelet_scaler.fit_transform(X_train_wavelet)
    X_test_wavelet_scaled = wavelet_scaler.transform(X_test_wavelet)

    # 7. 创建和编译WNN模型
    print("正在创建小波神经网络模型...")
    model = create_wnn_model(
        input_shape=(X_train_scaled.shape[1],),
        wavelet_features_shape=(X_train_wavelet_scaled.shape[1],),
        num_classes=num_classes
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("模型结构:")
    model.summary()

    # 8. 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 9. 训练模型
    print("正在进行小波神经网络模型训练...")
    history = model.fit(
        [X_train_scaled, X_train_wavelet_scaled], 
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 10. 模型评估
    print("正在进行模型评估...")
    y_pred_proba = model.predict([X_test_scaled, X_test_wavelet_scaled])
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- 模型评估结果 (WNN - 无特征选择) ---")
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    
    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=[str(label) for label in le.classes_]))
    
    # 显示特征使用信息
    print(f"\n特征使用信息:")
    print(f"原始特征数量: {X_train_scaled.shape[1]} (全部特征)")
    print(f"小波特征数量: {X_train_wavelet_scaled.shape[1]}")
    print(f"总特征数量: {X_train_scaled.shape[1] + X_train_wavelet_scaled.shape[1]}")
    print(f"特征选择方法: 无")
    print(f"模型类型: 小波神经网络(WNN)")
    print(f"使用小波: db4")
    print(f"分解层数: 3")
    print(f"训练轮数: {len(history.history['loss'])}")
    
    # 显示训练历史
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n训练历史:")
    print(f"最终训练损失: {final_train_loss:.4f}")
    print(f"最终验证损失: {final_val_loss:.4f}")
    print(f"最终训练准确率: {final_train_acc:.4f}")
    print(f"最终验证准确率: {final_val_acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_features': X_train_scaled.shape[1],
        'n_wavelet_features': X_train_wavelet_scaled.shape[1],
        'total_features': X_train_scaled.shape[1] + X_train_wavelet_scaled.shape[1],
        'method': 'WNN-无特征选择',
        'model': model,
        'scaler': scaler,
        'wavelet_scaler': wavelet_scaler,
        'label_encoder': le,
        'history': history,
        'num_classes': num_classes
    }

if __name__ == '__main__':
    # 配置GPU
    gpu_available = configure_gpu()
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("📈 开始小波神经网络无特征选择实验...")
    if gpu_available:
        print("🚀 使用GPU进行训练")
    
    try:
        results = process_spectrum_data(data_path)
        print(f"\n🎉 小波神经网络实验完成!")
        print(f"最终模型性能: {results['accuracy']:.2f}%")
        print(f"原始特征: {results['n_features']}")
        print(f"小波特征: {results['n_wavelet_features']}")
        print(f"总特征数: {results['total_features']}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}") 