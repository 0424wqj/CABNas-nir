import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, classification_report
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import autokeras as ak
import tensorflow as tf
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
            print("✅ GPU配置成功，启用内存动态增长")
            
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

class BaselineDriftAugmentation:
    """基线漂移数据增强器"""
    
    def __init__(self, random_state=42):
        np.random.seed(random_state)
        self.random_state = random_state
    
    def add_baseline_drift(self, spectrum, drift_strength=0.05):
        """添加基线漂移"""
        n_points = len(spectrum)
        
        # 确保有足够的控制点进行插值
        min_points = max(4, min(6, n_points // 10))
        max_points = min(10, n_points // 5)
        drift_points = np.random.randint(min_points, max_points + 1)
        
        if drift_points >= n_points:
            drift_points = n_points - 1
            
        # 总是包含首尾两个点以确保边界条件
        control_indices = [0, n_points - 1]
        
        if drift_points > 2:
            middle_points = np.random.choice(
                range(1, n_points - 1), 
                size=drift_points - 2, 
                replace=False
            )
            control_indices.extend(middle_points)
        
        control_x = np.sort(control_indices)
        control_y = np.random.normal(0, drift_strength, len(control_x))
        
        # 确保边界点的漂移较小
        control_y[0] *= 0.5
        control_y[-1] *= 0.5
        
        try:
            if len(control_x) >= 4:
                f = interp1d(control_x, control_y, kind='cubic', 
                            bounds_error=False, fill_value=0)
            else:
                f = interp1d(control_x, control_y, kind='linear', 
                            bounds_error=False, fill_value=0)
            
            baseline = f(np.arange(n_points))
            
            if np.any(np.isnan(baseline)) or np.any(np.isinf(baseline)):
                baseline = np.linspace(control_y[0], control_y[-1], n_points)
                
        except Exception as e:
            print(f"⚠️ 插值失败，使用简单漂移: {e}")
            baseline = np.random.normal(0, drift_strength * 0.5, n_points)
        
        return spectrum + baseline
    
    def augment_data(self, X, y, augmentation_factor=3, drift_strength=0.05):
        """对数据进行基线漂移增强"""
        print(f"📈 正在进行基线漂移数据增强...")
        print(f"   增强倍数: {augmentation_factor}")
        print(f"   漂移强度: {drift_strength}")
        
        X_augmented_list = [X]
        y_augmented_list = [y]
        
        for i in range(augmentation_factor):
            X_aug = np.array([self.add_baseline_drift(spectrum, drift_strength) for spectrum in X])
            X_augmented_list.append(X_aug)
            y_augmented_list.append(y)
        
        X_augmented = np.vstack(X_augmented_list)
        y_augmented = np.hstack(y_augmented_list)
        
        print(f"   原始数据量: {len(X)} -> 增强后数据量: {len(X_augmented)}")
        
        return X_augmented, y_augmented

class ConsistencyActiveLearning:
    """Consistency-based 主动学习采样器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_diverse_models(self, X_train, y_train, n_models=5):
        """
        创建多个不同的基础模型用于一致性检查
        包括不同类型的分类器以增加多样性
        """
        print(f"🎯 创建 {n_models} 个多样化模型用于一致性评估")
        
        models = []
        model_names = []
        
        try:
            # 1. 随机森林 (不同参数)
            rf1 = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10, 
                random_state=self.random_state,
                n_jobs=-1
            )
            models.append(rf1)
            model_names.append("RandomForest_1")
            
            rf2 = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15,
                min_samples_split=5,
                random_state=self.random_state + 1,
                n_jobs=-1
            )
            models.append(rf2)
            model_names.append("RandomForest_2")
            
            # 2. SVM (不同核函数)
            if len(X_train) < 1000:  # SVM对大数据集较慢
                svm_rbf = SVC(
                    kernel='rbf', 
                    probability=True,
                    random_state=self.random_state,
                    C=1.0
                )
                models.append(svm_rbf)
                model_names.append("SVM_RBF")
                
                svm_linear = SVC(
                    kernel='linear', 
                    probability=True,
                    random_state=self.random_state,
                    C=0.1
                )
                models.append(svm_linear)
                model_names.append("SVM_Linear")
            
            # 3. 神经网络 (不同架构)
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500,
                alpha=0.01
            )
            models.append(mlp)
            model_names.append("MLP")
            
            # 训练所有模型
            trained_models = []
            successful_names = []
            
            for i, (model, name) in enumerate(zip(models, model_names)):
                try:
                    print(f"   训练模型 {i+1}/{len(models)}: {name}")
                    model.fit(X_train, y_train)
                    trained_models.append(model)
                    successful_names.append(name)
                except Exception as e:
                    print(f"   ⚠️ 模型 {name} 训练失败: {e}")
                    continue
            
            print(f"   成功训练 {len(trained_models)} 个模型")
            return trained_models, successful_names
            
        except Exception as e:
            print(f"⚠️ 创建多样化模型失败: {e}")
            # 回退到单一随机森林模型
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            rf.fit(X_train, y_train)
            return [rf], ["RandomForest_Fallback"]
    
    def compute_prediction_consistency(self, models, X_pool):
        """
        计算多个模型对候选样本预测的一致性
        一致性低的样本更有价值，因为它们在不同模型间产生分歧
        """
        print(f"🎯 计算 {len(models)} 个模型的预测一致性")
        
        try:
            # 收集所有模型的预测
            all_predictions = []
            
            for i, model in enumerate(models):
                try:
                    if hasattr(model, 'predict_proba'):
                        # 使用概率预测
                        proba = model.predict_proba(X_pool)
                        pred = np.argmax(proba, axis=1)
                    else:
                        pred = model.predict(X_pool)
                    all_predictions.append(pred)
                except Exception as e:
                    print(f"   ⚠️ 模型 {i} 预测失败: {e}")
                    continue
            
            if len(all_predictions) == 0:
                print("⚠️ 所有模型预测都失败")
                return np.zeros(len(X_pool))
            
            all_predictions = np.array(all_predictions)
            
            # 计算每个样本的一致性分数
            consistency_scores = []
            
            for i in range(len(X_pool)):
                sample_predictions = all_predictions[:, i]
                
                # 计算一致性：预测相同的模型数量比例
                unique_preds, counts = np.unique(sample_predictions, return_counts=True)
                max_agreement = np.max(counts)
                consistency = max_agreement / len(all_predictions)
                
                # 我们要选择一致性低的样本（分歧大的样本）
                inconsistency_score = 1.0 - consistency
                consistency_scores.append(inconsistency_score)
            
            consistency_scores = np.array(consistency_scores)
            
            # 统计信息
            avg_consistency = 1.0 - np.mean(consistency_scores)
            min_consistency = 1.0 - np.max(consistency_scores)
            max_consistency = 1.0 - np.min(consistency_scores)
            
            print(f"   平均一致性: {avg_consistency:.4f}")
            print(f"   最小一致性: {min_consistency:.4f}")
            print(f"   最大一致性: {max_consistency:.4f}")
            print(f"   一致性标准差: {np.std(1.0 - consistency_scores):.4f}")
            
            return consistency_scores
            
        except Exception as e:
            print(f"⚠️ 一致性计算失败: {e}")
            return np.random.random(len(X_pool))  # 回退到随机分数
    
    def consistency_sampling(self, models, X_pool, n_samples=50):
        """
        Consistency-based 采样策略
        选择多个模型预测一致性最低的样本（分歧最大的样本）
        """
        print(f"🎯 执行Consistency采样，选择 {n_samples} 个样本")
        
        try:
            # 计算一致性分数
            inconsistency_scores = self.compute_prediction_consistency(models, X_pool)
            
            # 选择不一致性最高的样本（一致性最低的样本）
            selected_indices = np.argsort(inconsistency_scores)[-n_samples:]
            
            # 统计信息
            avg_inconsistency = np.mean(inconsistency_scores[selected_indices])
            max_inconsistency = np.max(inconsistency_scores[selected_indices])
            min_inconsistency = np.min(inconsistency_scores[selected_indices])
            
            print(f"   平均不一致性: {avg_inconsistency:.4f}")
            print(f"   最大不一致性: {max_inconsistency:.4f}")
            print(f"   最小不一致性: {min_inconsistency:.4f}")
            print(f"   不一致性标准差: {np.std(inconsistency_scores[selected_indices]):.4f}")
            
            return selected_indices, inconsistency_scores[selected_indices]
            
        except Exception as e:
            print(f"⚠️ Consistency采样失败: {e}")
            # 回退到随机采样
            return np.random.choice(len(X_pool), n_samples, replace=False), np.array([])
    
    def active_learning_selection(self, X_augmented, y_augmented, n_samples=200, 
                                 validation_split=0.2):
        """
        执行Consistency-based主动学习样本选择
        """
        print(f"\n🎯 开始Consistency-based主动学习样本选择")
        print(f"总样本数: {len(X_augmented)}")
        print(f"目标选择数: {n_samples}")
        
        # 1. 初始训练集分割
        n_initial = min(100, len(X_augmented) // 4)  # 初始训练集大小
        initial_indices = np.random.choice(len(X_augmented), n_initial, replace=False)
        
        X_initial = X_augmented[initial_indices]
        y_initial = y_augmented[initial_indices]
        
        # 剩余样本作为候选池
        remaining_indices = np.array([i for i in range(len(X_augmented)) if i not in initial_indices])
        X_pool = X_augmented[remaining_indices]
        y_pool = y_augmented[remaining_indices]
        
        print(f"初始训练集: {len(X_initial)} 样本")
        print(f"候选池: {len(X_pool)} 样本")
        
        # 2. 创建多样化模型集合
        models, model_names = self.create_diverse_models(X_initial, y_initial)
        print(f"成功创建模型: {model_names}")
        
        # 3. 执行Consistency采样
        if len(X_pool) <= n_samples - len(X_initial):
            # 如果候选池样本不足，全部选择
            selected_pool_indices = np.arange(len(X_pool))
            selected_scores = np.array([])
        else:
            selected_pool_indices, selected_scores = self.consistency_sampling(
                models, X_pool, n_samples - len(X_initial)
            )
        
        # 4. 组合最终选择的样本
        final_indices = np.concatenate([
            initial_indices,
            remaining_indices[selected_pool_indices]
        ])
        
        X_selected = X_augmented[final_indices]
        y_selected = y_augmented[final_indices]
        
        print(f"\n✅ Consistency主动学习选择完成")
        print(f"最终选择样本数: {len(X_selected)}")
        print(f"选择率: {len(X_selected)/len(X_augmented)*100:.1f}%")
        
        return X_selected, y_selected, final_indices, selected_scores

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
                
    except Exception as e:
        print(f"❌ 打印架构时出错: {str(e)}")
    
    print("="*70)

def process_spectrum_data_with_consistency_active_learning(file_path):
    """
    CARS特征选择 + 基线漂移数据增强 + Consistency主动学习 + AutoKeras CNN+RNN
    """
    print("="*80)
    print("   CARS + 基线漂移增强 + Consistency主动学习 + AutoKeras CNN+RNN")
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
        X_sg, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
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

    # 7. 基线漂移数据增强
    print("\n步骤7: 正在进行基线漂移数据增强...")
    augmenter = BaselineDriftAugmentation(random_state=42)
    X_train_augmented, y_train_augmented = augmenter.augment_data(
        X_train_cars, y_train, 
        augmentation_factor=3,
        drift_strength=0.05
    )

    # 8. Consistency主动学习采样
    print("\n步骤8: 正在进行Consistency主动学习采样...")
    active_learner = ConsistencyActiveLearning(random_state=42)
    X_selected, y_selected, selected_indices, scores = active_learner.active_learning_selection(
        X_train_augmented, y_train_augmented,
        n_samples=min(400, len(X_train_augmented) // 2),  # 选择一半样本
        validation_split=0.2
    )

    # 9. 调整数据形状以适应CNN+RNN
    print("\n步骤9: 正在调整数据形状...")
    X_train_reshaped = X_selected.reshape(X_selected.shape[0], 1, X_selected.shape[1])
    X_test_reshaped = X_test_cars.reshape(X_test_cars.shape[0], 1, X_test_cars.shape[1])
    y_train_final = y_selected.astype(np.int32)
    y_test_final = y_test.astype(np.int32)
    
    print(f"训练数据形状: {X_train_reshaped.shape}")
    print(f"测试数据形状: {X_test_reshaped.shape}")

    # 10. 创建AutoKeras模型
    print("\n步骤10: 正在创建AutoKeras CNN+RNN模型...")
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
    
    # 11. 训练模型
    model.fit(
        X_train_reshaped, 
        y_train_final,
        validation_split=0.2,
        epochs=800,
        verbose=1
    )
    
    print("AutoKeras模型训练完成！")
    
    # 打印网络结构
    print_model_architecture(model, "CARS + 基线漂移增强 + Consistency主动学习 + AutoKeras CNN+RNN")
    
    # 12. 模型评估
    print("\n步骤11: 正在进行模型评估...")
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
    print("  CARS + 基线漂移增强 + Consistency主动学习 + AutoKeras 模型评估结果")
    print("="*80)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精准率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1值: {f1:.2f}%")
    print("="*80)
    
    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(y_test_final, y_pred_final, target_names=[str(label) for label in le.classes_]))
    
    # 显示主动学习效果总结
    print(f"\n🎯 Consistency主动学习效果总结:")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"CARS选择特征: {X_train_cars.shape[1]} ({X_train_cars.shape[1]/X.shape[1]*100:.1f}%)")
    print(f"增强后总样本: {len(X_train_augmented)}")
    print(f"主动学习选择: {len(X_selected)} ({len(X_selected)/len(X_train_augmented)*100:.1f}%)")
    print(f"数据压缩率: {len(X_selected)/len(X_train_augmented):.2f}")
    print(f"最终模型性能: {accuracy:.2f}%")
    if len(scores) > 0:
        print(f"选择样本平均不一致性: {np.mean(scores):.4f}")
    
    # 获取最终模型
    exported_model = model.export_model()
    
    return {
        'method': 'Consistency主动学习',
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
        'augmented_samples': len(X_train_augmented),
        'selected_samples': len(X_selected),
        'selection_ratio': len(X_selected)/len(X_train_augmented),
        'avg_inconsistency': np.mean(scores) if len(scores) > 0 else 0
    }

if __name__ == '__main__':
    # 配置GPU
    gpu_available = configure_gpu()
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    data_path = r'C:\Users\Administrator\Desktop\管道淤泥项目\光谱\近红外数据\4.1数据-近红外\65℃-过筛\65烘干过筛.csv'
    
    print("📈 启动CARS + 基线漂移增强 + Consistency主动学习 + AutoKeras CNN+RNN实验...")
    if gpu_available:
        print("🚀 使用GPU加速训练")
    
    try:
        result = process_spectrum_data_with_consistency_active_learning(data_path)
        
        print(f"\n🎉 Consistency主动学习实验完成！")
        print(f"最终模型性能: {result['accuracy']:.2f}%")
        print(f"数据选择率: {result['selection_ratio']:.2f}")
        print(f"样本压缩: {result['augmented_samples']} -> {result['selected_samples']}")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}") 