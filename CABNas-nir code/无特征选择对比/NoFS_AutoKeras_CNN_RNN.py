import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.signal import savgol_filter
import autokeras as ak
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

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
                
        else:
            print("⚠️ 无法获取详细架构信息 - 模型可能尚未训练完成")
            
    except Exception as e:
        print(f"❌ 打印架构时出错: {str(e)}")
        print("这可能是由于模型尚未完全训练完成")
    
    print("="*70)

def process_spectrum_data(file_path):
    """
    不使用特征选择的AutoKeras CNN+RNN分类
    """
    print("="*70)
    print("           无特征选择 - AutoKeras CNN+RNN分类")
    print("="*70)
    
    # 1. 加载数据
    print("正在加载数据...")
    data = pd.read_csv(file_path)
    data = data.dropna()
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    print(f"原始数据形状: {X.shape}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

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
        X_sg, y_encoded, test_size=0.3, random_state=44, stratify=y_encoded
    )

    # 5. 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"使用全部特征数量: {X_train_scaled.shape[1]}")

    # 6. 调整数据形状以适应CNN+RNN (samples, timesteps, features)
    print("正在调整数据形状...")
    # 将数据reshape为3D: (samples, 1, features) 适合1D卷积处理
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    
    # 确保标签为正确的形状和类型
    y_train_final = y_train.astype(np.int32)
    y_test_final = y_test.astype(np.int32)
    
    print(f"训练数据形状: {X_train_reshaped.shape}")
    print(f"测试数据形状: {X_test_reshaped.shape}")
    print(f"训练标签形状: {y_train_final.shape}")
    print(f"测试标签形状: {y_test_final.shape}")

    # 7. 创建简化的AutoKeras CNN+RNN模型（使用全部特征）
    print("正在创建简化的AutoKeras CNN+RNN模型（使用全部特征）...")
    
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
    print_model_architecture(autokeras_model, "无特征选择 + AutoKeras CNN+RNN 搜索结果")
    
    # 8. 模型评估
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

    print("\n" + "="*70)
    print("     无特征选择 + AutoKeras CNN+RNN 模型评估结果")
    print("="*70)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    
    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(y_test_final, y_pred_final, target_names=[str(label) for label in le.classes_]))
    print("="*70)
    
    # 显示特征使用信息
    print(f"\n特征使用信息:")
    print(f"使用特征数量: {X_train_scaled.shape[1]} (全部特征)")
    print(f"特征选择方法: 无")
    print(f"输入形状: {X_train_reshaped.shape[1:]}")
    print(f"类别数量: {len(le.classes_)}")
    print(f"架构: 简化CNN(特征提取) + LSTM(序列建模) + 分类头")
    
    # 获取最终模型
    print("\n正在导出最佳模型...")
    final_model = autokeras_model.export_model()
    print("模型导出成功！")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_features': X_train_scaled.shape[1],
        'method': 'AutoKeras CNN+RNN-无特征选择',
        'model': final_model,
        'scaler': scaler,
        'label_encoder': le
    }

if __name__ == '__main__':
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("正在启动无特征选择的AutoKeras CNN+RNN分类流程...")
    
    result = process_spectrum_data(data_path)
    
    if result['model'] is not None:
        print("\n流程执行成功！")
        print("已生成：")
        print("1. 训练好的AutoKeras CNN+RNN模型")
        print("2. 数据标准化器")
        print("3. 标签编码器")
    else:
        print("\n流程执行遇到问题，请检查环境配置。") 