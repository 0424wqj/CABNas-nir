import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(input_shape, num_classes, embed_dim=128, num_heads=8, ff_dim=512, num_transformer_blocks=2):
    """创建Transformer分类模型"""
    inputs = Input(shape=input_shape)
    
    # 将1D光谱数据重塑为序列
    x = tf.expand_dims(inputs, axis=-1)  # (batch_size, seq_len, 1)
    
    # 位置编码和投影
    x = Dense(embed_dim)(x)
    
    # 添加Transformer块
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    # 全局平均池化
    x = GlobalAveragePooling1D()(x)
    
    # 分类头
    x = Dropout(0.3)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model

def process_spectrum_data(file_path):
    """
    不使用特征选择的Transformer分类
    """
    print("="*60)
    print("        无特征选择 - Transformer分类")
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
        X_sg, y_encoded, test_size=0.4, random_state=44, stratify=y_encoded
    )

    # 5. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"使用全部特征数量: {X_train_scaled.shape[1]}")
    print(f"训练集形状: {X_train_scaled.shape}")
    print(f"测试集形状: {X_test_scaled.shape}")

    # 6. 创建和编译Transformer模型
    print("正在创建Transformer模型...")
    model = create_transformer_model(
        input_shape=(X_train_scaled.shape[1],),
        num_classes=num_classes,
        embed_dim=128,
        num_heads=8,
        ff_dim=512,
        num_transformer_blocks=3
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("模型结构:")
    model.summary()

    # 7. 设置回调函数
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

    # 8. 训练模型
    print("正在进行Transformer模型训练...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 9. 模型评估
    print("正在进行模型评估...")
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- 模型评估结果 (Transformer - 无特征选择) ---")
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    
    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=[str(label) for label in le.classes_]))
    
    # 显示特征使用信息
    print(f"\n特征使用信息:")
    print(f"使用特征数量: {X_train_scaled.shape[1]} (全部特征)")
    print(f"特征选择方法: 无")
    print(f"模型类型: Transformer")
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
        'method': 'Transformer-无特征选择',
        'model': model,
        'scaler': scaler,
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
    
    print("📈 开始Transformer无特征选择实验...")
    if gpu_available:
        print("🚀 使用GPU进行训练")
    
    try:
        results = process_spectrum_data(data_path)
        print(f"\n🎉 Transformer实验完成!")
        print(f"最终模型性能: {results['accuracy']:.2f}%")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}") 