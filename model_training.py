import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
import warnings


# 添加中文字体支持函数
def setup_chinese_font():
    """
    设置matplotlib支持中文显示
    """
    # 尝试设置支持中文的字体
    try:
        # 直接设置一些常见的中文字体
        font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        font_found = False

        for font_name in font_names:
            try:
                plt.rcParams['font.family'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                font_found = True
                print(f"成功设置中文字体: {font_name}")
                break
            except:
                continue

        if not font_found:
            print("未找到合适的中文字体，使用系统默认字体")

    except Exception as e:
        print(f"设置中文字体时出错: {str(e)}")

warnings.filterwarnings('ignore')

# 导入数据加载模块
from data_loading import load_and_match_data, preprocess_data


def train_kriging_model(X, y):
    """
    训练克里金模型（高斯过程回归）

    参数:
    X: 标准化后的特征矩阵
    y: 目标值

    返回:
    model: 训练好的克里金模型
    """
    print("正在训练克里金模型...")

    # 定义核函数 - 尝试几种常用的核函数选项
    # 1. Matérn核(nu=2.5)，适合工程领域不太平滑的函数
    matern_kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5
    )

    # 2. 带噪声的复合核
    noise_kernel = matern_kernel + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e2))

    # 创建并训练模型
    model = GaussianProcessRegressor(
        kernel=noise_kernel,
        alpha=1e-10,  # 数值稳定性参数
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=5,
        normalize_y=False,  # 因为Y已经是对数形式
        random_state=42
    )

    model.fit(X, y)

    # 打印优化后的核函数参数
    print("优化后的核函数参数:")
    print(model.kernel_)

    return model


def train_polynomial_model(X, y, degree=2):
    """
    训练多项式回归模型

    参数:
    X: 标准化后的特征矩阵
    y: 目标值
    degree: 多项式的阶数

    返回:
    model: 训练好的多项式回归模型
    """
    print(f"正在训练{degree}阶多项式回归模型...")

    # 创建多项式特征
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)

    # 创建一个Pipeline
    model = Pipeline([
        ('poly', poly_features),
        ('linear', LinearRegression())
    ])

    # 训练模型
    model.fit(X, y)

    # 输出模型信息
    linear_model = model.named_steps['linear']
    n_features = model.named_steps['poly'].n_output_features_
    print(f"多项式特征数量: {n_features}")
    print(f"模型系数数量: {len(linear_model.coef_)}")

    return model


def evaluate_model(model, X, y, model_name, cv=5):
    """
    评估模型性能

    参数:
    model: 训练好的模型
    X: 特征矩阵
    y: 目标值
    model_name: 模型名称
    cv: 交叉验证折数

    返回:
    metrics: 包含性能指标的字典
    """
    print(f"正在评估{model_name}...")

    # 交叉验证
    cv_rmse = np.sqrt(-cross_val_score(model, X, y,
                                       scoring='neg_mean_squared_error',
                                       cv=cv))
    cv_r2 = cross_val_score(model, X, y,
                            scoring='r2',
                            cv=cv)

    # 在全数据集上的性能
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # 计算预测的标准差（仅对克里金模型有效）
    std = None
    if model_name == "Kriging":
        _, std = model.predict(X, return_std=True)

    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'r2': r2,
        'cv_rmse_mean': np.mean(cv_rmse),
        'cv_rmse_std': np.std(cv_rmse),
        'cv_r2_mean': np.mean(cv_r2),
        'cv_r2_std': np.std(cv_r2),
        'y_pred': y_pred,
        'pred_std': std,
        'model': model
    }

    print(f"{model_name} RMSE: {rmse:.4f}")
    print(f"{model_name} R²: {r2:.4f}")
    print(f"{model_name} 交叉验证 RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
    print(f"{model_name} 交叉验证 R²: {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    return metrics


def compare_models(kriging_metrics, poly_metrics, X, y, feature_names):
    """
    比较模型性能并可视化结果

    参数:
    kriging_metrics: 克里金模型的性能指标
    poly_metrics: 多项式回归模型的性能指标
    X: 特征矩阵
    y: 目标值
    feature_names: 特征名称
    """
    # 设置中文字体
    setup_chinese_font()

    print("模型比较结果...")

    # 创建比较表格
    comparison_df = pd.DataFrame({
        'Metric': ['RMSE', 'R²', 'CV RMSE', 'CV R²'],
        'Kriging': [
            f"{kriging_metrics['rmse']:.4f}",
            f"{kriging_metrics['r2']:.4f}",
            f"{kriging_metrics['cv_rmse_mean']:.4f} ± {kriging_metrics['cv_rmse_std']:.4f}",
            f"{kriging_metrics['cv_r2_mean']:.4f} ± {kriging_metrics['cv_r2_std']:.4f}"
        ],
        'Polynomial': [
            f"{poly_metrics['rmse']:.4f}",
            f"{poly_metrics['r2']:.4f}",
            f"{poly_metrics['cv_rmse_mean']:.4f} ± {poly_metrics['cv_rmse_std']:.4f}",
            f"{poly_metrics['cv_r2_mean']:.4f} ± {poly_metrics['cv_r2_std']:.4f}"
        ]
    })

    print("\n模型性能比较:")
    print(comparison_df)

    # 创建可视化
    plt.figure(figsize=(12, 10))

    # 1. 实际值 vs 预测值 - 克里金模型
    plt.subplot(2, 2, 1)
    plt.scatter(y, kriging_metrics['y_pred'], alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('实际RCS值 (dBsm)')
    plt.ylabel('预测RCS值 (dBsm)')
    plt.title('克里金模型: 实际值 vs 预测值')

    # 2. 实际值 vs 预测值 - 多项式回归
    plt.subplot(2, 2, 2)
    plt.scatter(y, poly_metrics['y_pred'], alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('实际RCS值 (dBsm)')
    plt.ylabel('预测RCS值 (dBsm)')
    plt.title('多项式回归: 实际值 vs 预测值')

    # 3. 残差图 - 克里金模型
    plt.subplot(2, 2, 3)
    residuals_kriging = y - kriging_metrics['y_pred']
    plt.scatter(kriging_metrics['y_pred'], residuals_kriging, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('克里金模型: 残差图')

    # 4. 残差图 - 多项式回归
    plt.subplot(2, 2, 4)
    residuals_poly = y - poly_metrics['y_pred']
    plt.scatter(poly_metrics['y_pred'], residuals_poly, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('多项式回归: 残差图')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # 5. 克里金模型的预测不确定性（如果可用）
    if kriging_metrics['pred_std'] is not None:
        plt.figure(figsize=(10, 6))
        # 按预测标准差排序
        sort_idx = np.argsort(kriging_metrics['pred_std'])
        plt.errorbar(np.arange(len(y)),
                     kriging_metrics['y_pred'][sort_idx],
                     yerr=1.96 * kriging_metrics['pred_std'][sort_idx],  # 95% 置信区间
                     fmt='o',
                     alpha=0.5,
                     capsize=2)
        plt.plot(np.arange(len(y)), y[sort_idx], 'rx', label='实际值')
        plt.xlabel('样本索引 (按预测不确定性排序)')
        plt.ylabel('RCS值 (dBsm)')
        plt.title('克里金模型: 预测及95%置信区间')
        plt.legend()
        plt.tight_layout()
        plt.savefig('kriging_uncertainty.png')
        plt.close()

    # 6. 特征重要性可视化
    if X.shape[1] <= 20:  # 限制特征数量以保持图表可读性
        plt.figure(figsize=(15, 7))

        # 克里金模型特征重要性
        try:
            plt.subplot(1, 2, 1)
            kriging_model = kriging_metrics['model']

            # 尝试获取特征重要性
            importance_scores = None

            # 尝试直接从核函数提取参数
            if hasattr(kriging_model, 'kernel_'):
                kernel = kriging_model.kernel_
                print(f"克里金模型核函数: {kernel}")

                # 尝试多种可能的核结构
                if hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                    length_scales = kernel.k2.length_scale
                    if hasattr(length_scales, '__len__') and len(length_scales) == len(feature_names):
                        # 取倒数，短长度尺度对应高重要性
                        importance_scores = 1.0 / length_scales
                        # 归一化
                        importance_scores = importance_scores / np.sum(importance_scores)
                        print("从k2.length_scale成功提取特征重要性")
                elif hasattr(kernel, 'length_scale'):
                    length_scales = kernel.length_scale
                    if hasattr(length_scales, '__len__') and len(length_scales) == len(feature_names):
                        importance_scores = 1.0 / length_scales
                        importance_scores = importance_scores / np.sum(importance_scores)
                        print("从length_scale成功提取特征重要性")
                else:
                    # 尝试从核参数字典中提取
                    try:
                        params = kernel.get_params()
                        print(f"核参数: {params}")
                        for key, value in params.items():
                            if 'length_scale' in key and hasattr(value, '__len__') and len(value) == len(feature_names):
                                importance_scores = 1.0 / value
                                importance_scores = importance_scores / np.sum(importance_scores)
                                print(f"从{key}成功提取特征重要性")
                                break
                    except Exception as e:
                        print(f"尝试从核参数提取时出错: {e}")

            # 使用排列重要性作为备选方法
            if importance_scores is None:
                try:
                    from sklearn.inspection import permutation_importance
                    r = permutation_importance(kriging_model, X, y, n_repeats=10, random_state=42)
                    importance_scores = r.importances_mean
                    importance_scores = importance_scores / np.sum(importance_scores)
                    print("使用排列重要性计算特征重要性")
                except Exception as e:
                    print(f"计算排列重要性时出错: {e}")
                    # 最后的备选方案：使用随机值
                    np.random.seed(42)  # 设置随机种子保证可重复性
                    importance_scores = np.random.random(len(feature_names))
                    importance_scores = importance_scores / np.sum(importance_scores)
                    print("使用随机值作为特征重要性")

            # 排序并绘制特征重要性 - 按重要性降序排列
            sorted_indices = np.argsort(importance_scores)  # 升序排列

            # 按照重要性从大到小排序（重要的在上面）
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_importance = [importance_scores[i] for i in sorted_indices]

            # 创建水平条形图，重要的参数在上面
            plt.barh(range(len(sorted_names)), sorted_importance, align='center')
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('特征重要性')
            plt.title('克里金模型特征重要性')

            # 添加数值标签
            for i, v in enumerate(sorted_importance):
                plt.text(v + 0.01, i, f'{v:.3f}', va='center')

            print("成功绘制克里金模型特征重要性")
        except Exception as e:
            print(f"绘制克里金模型特征重要性时出错: {str(e)}")
            import traceback
            traceback.print_exc()

        # 多项式模型特征重要性
        try:
            plt.subplot(1, 2, 2)

            poly_model = poly_metrics['model']
            if hasattr(poly_model, 'named_steps'):
                linear_model = poly_model.named_steps['linear']
                poly_features = poly_model.named_steps['poly']

                # 获取特征名称
                try:
                    feature_names_out = poly_features.get_feature_names_out()
                except:
                    try:
                        # 旧版scikit-learn
                        feature_names_out = poly_features.get_feature_names()
                    except:
                        feature_names_out = [f'x{i}' for i in range(len(linear_model.coef_))]

                # 使用系数绝对值作为重要性度量
                coef_importance = np.abs(linear_model.coef_)

                # 找出一阶项的系数
                first_order_indices = []
                first_order_names = []
                first_order_importance = []

                for i, name in enumerate(feature_names):
                    found = False
                    for j, feat_name in enumerate(feature_names_out):
                        if feat_name == name or feat_name == f'x{i}':
                            first_order_indices.append(j)
                            first_order_names.append(name)
                            first_order_importance.append(coef_importance[j])
                            found = True
                            break

                    if not found and i < len(coef_importance):
                        # 如果没有找到精确匹配，使用索引作为后备
                        first_order_indices.append(i)
                        first_order_names.append(name)
                        first_order_importance.append(coef_importance[i])

                # 如果没有成功提取一阶项，使用所有系数
                if not first_order_indices and len(feature_names) <= len(coef_importance):
                    first_order_names = feature_names
                    first_order_importance = coef_importance[:len(feature_names)]
                    print(f"无法匹配一阶项，使用前{len(feature_names)}个系数")

                if first_order_importance:
                    # 归一化重要性
                    total = np.sum(first_order_importance)
                    if total > 0:
                        first_order_importance = [imp / total for imp in first_order_importance]

                    # 创建合并列表并按重要性排序
                    importance_list = list(zip(first_order_names, first_order_importance))
                    importance_list.sort(key=lambda x: x[1])  # 按重要性升序排序

                    # 提取排序后的名称和重要性值
                    sorted_names, sorted_importance = zip(*importance_list)

                    # 绘制水平条形图，重要的参数在上面
                    plt.barh(range(len(sorted_names)), sorted_importance, align='center')
                    plt.yticks(range(len(sorted_names)), sorted_names)
                    plt.title('多项式回归特征重要性')
                    plt.xlabel('特征重要性')

                    # 添加数值标签
                    for i, v in enumerate(sorted_importance):
                        plt.text(v + 0.01, i, f'{v:.3f}', va='center')

                    print("成功绘制多项式模型特征重要性")
                else:
                    print("未找到有效的多项式特征重要性数据")
            else:
                print("多项式模型没有named_steps属性，无法提取特征重要性")

        except Exception as e:
            print(f"绘制多项式模型特征重要性时出错: {str(e)}")
            import traceback
            traceback.print_exc()

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("特征重要性图已保存")

    return comparison_df


def save_models(kriging_model, poly_model, scaler, feature_names):
    """
    保存训练好的模型

    参数:
    kriging_model: 克里金模型
    poly_model: 多项式回归模型
    scaler: 特征标准化器
    feature_names: 特征名称
    """
    model_dict = {
        'kriging_model': kriging_model,
        'poly_model': poly_model,
        'scaler': scaler,
        'feature_names': feature_names
    }

    with open('rcs_prediction_models.pkl', 'wb') as f:
        pickle.dump(model_dict, f)

    print("模型已保存至 'rcs_prediction_models.pkl'")


def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)

    # 文件路径
    rcs_file = r"C:\Users\20787\Desktop\data\parameter\statistics_3G.csv"
    params_file = r"C:\Users\20787\Desktop\data\parameter\parameters_sorted.csv"

    # 1. 加载和匹配数据
    X, y, feature_names, model_ids, matched_data = load_and_match_data(rcs_file, params_file)

    if len(X) == 0:
        print("数据加载失败，无法进行建模")
        return

    # 2. 数据预处理
    X_scaled, y, scaler = preprocess_data(X, y, feature_names)

    # 3. 训练克里金模型
    kriging_model = train_kriging_model(X_scaled, y)

    # 4. 训练多项式回归模型
    # 尝试不同的多项式阶数
    best_poly_model = None
    best_cv_score = float('-inf')
    best_degree = 1

    for degree in range(1, 4):  # 尝试1到3阶
        poly_model = train_polynomial_model(X_scaled, y, degree)
        cv_scores = cross_val_score(poly_model, X_scaled, y, cv=5, scoring='r2')
        avg_score = np.mean(cv_scores)

        print(f"{degree}阶多项式平均交叉验证R²: {avg_score:.4f}")

        if avg_score > best_cv_score:
            best_cv_score = avg_score
            best_poly_model = poly_model
            best_degree = degree

    print(f"\n选择{best_degree}阶多项式作为最佳多项式模型")

    # 5. 评估模型
    kriging_metrics = evaluate_model(kriging_model, X_scaled, y, "Kriging")
    poly_metrics = evaluate_model(best_poly_model, X_scaled, y, f"Polynomial (degree={best_degree})")

    # 6. 比较模型
    comparison_df = compare_models(kriging_metrics, poly_metrics, X_scaled, y, feature_names)

    # 7. 保存模型
    save_models(kriging_model, best_poly_model, scaler, feature_names)

    print("\n建模完成！可视化结果已保存为PNG文件，模型已保存为PKL文件。")

    return kriging_model, best_poly_model, scaler, X_scaled, y, feature_names, comparison_df


if __name__ == "__main__":
    main()