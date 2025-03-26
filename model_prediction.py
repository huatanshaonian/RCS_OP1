import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


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

def load_models(model_file='rcs_prediction_models.pkl'):
    """
    从磁盘加载已训练的模型

    参数:
    model_file: 模型文件路径

    返回:
    model_dict: 包含模型和相关组件的字典
    """
    try:
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)

        print(f"已成功加载模型: {model_file}")
        print(f"模型包含以下组件: {list(model_dict.keys())}")
        print(f"特征名称: {model_dict['feature_names']}")

        return model_dict
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None


def predict_rcs(design_params, model_dict, model_type='kriging'):
    """
    预测新设计参数的RCS值

    参数:
    design_params: 包含设计参数的字典或数组
    model_dict: 包含模型和组件的字典
    model_type: 'kriging' 或 'poly'

    返回:
    rcs_value: 预测的RCS值
    uncertainty: 预测的不确定性（仅克里金模型）
    """
    # 提取模型组件
    feature_names = model_dict['feature_names']
    scaler = model_dict['scaler']

    if model_type.lower() == 'kriging':
        model = model_dict['kriging_model']
    else:
        model = model_dict['poly_model']

    # 将设计参数转换为合适的格式
    if isinstance(design_params, dict):
        # 确保所有特征都存在
        param_values = []
        for name in feature_names:
            if name in design_params:
                param_values.append(design_params[name])
            else:
                print(f"警告: 设计参数中缺少特征 '{name}'，使用0代替")
                param_values.append(0)
        X_new = np.array([param_values])
    else:
        # 如果输入是数组，确保长度正确
        if len(design_params) != len(feature_names):
            print(f"警告: 设计参数数量 ({len(design_params)}) 与模型特征数量 ({len(feature_names)}) 不匹配")
        X_new = np.array([design_params])

    # 标准化特征
    X_new_scaled = scaler.transform(X_new)

    # 使用模型预测
    if model_type.lower() == 'kriging':
        # 克里金模型返回预测值和不确定性
        rcs_value, uncertainty = model.predict(X_new_scaled, return_std=True)
        return rcs_value[0], uncertainty[0]
    else:
        # 多项式模型只返回预测值
        rcs_value = model.predict(X_new_scaled)
        return rcs_value[0], None


def compare_predictions(design_params_list, model_dict, labels=None):
    """
    比较多个设计参数的RCS预测值

    参数:
    design_params_list: 设计参数列表
    model_dict: 模型字典
    labels: 设计标签列表

    返回:
    comparison_df: 比较结果的数据框
    """
    if labels is None:
        labels = [f"设计{i + 1}" for i in range(len(design_params_list))]

    results = []

    for i, params in enumerate(design_params_list):
        kriging_pred, kriging_unc = predict_rcs(params, model_dict, 'kriging')
        poly_pred, _ = predict_rcs(params, model_dict, 'poly')

        results.append({
            '设计': labels[i],
            'Kriging预测(dBsm)': kriging_pred,
            'Kriging不确定性': kriging_unc,
            'Polynomial预测(dBsm)': poly_pred,
            '模型差异': abs(kriging_pred - poly_pred)
        })

    comparison_df = pd.DataFrame(results)
    print("预测结果比较:")
    print(comparison_df)

    return comparison_df


def sensitivity_analysis(base_design, model_dict, n_points=20, model_type='kriging'):
    """
    进行设计参数敏感度分析

    参数:
    base_design: 基准设计参数
    model_dict: 模型字典
    n_points: 每个参数扫描的点数
    model_type: 使用的模型类型

    返回:
    sensitivity_results: 敏感度分析结果
    """
    feature_names = model_dict['feature_names']
    scaler = model_dict['scaler']

    # 获取特征均值和标准差
    feature_means = scaler.mean_
    feature_stds = np.sqrt(scaler.var_)

    # 将基准设计转换为数组
    if isinstance(base_design, dict):
        base_array = np.array([base_design.get(name, 0) for name in feature_names])
    else:
        base_array = np.array(base_design)

    # 存储结果
    sensitivity_results = {}

    # 对每个特征进行扫描
    for i, name in enumerate(feature_names):
        # 创建扫描范围 (±3个标准差)
        feature_min = max(0, base_array[i] - 3 * feature_stds[i])  # 防止小于0
        feature_max = base_array[i] + 3 * feature_stds[i]
        scan_values = np.linspace(feature_min, feature_max, n_points)

        # 初始化结果数组
        rcs_predictions = np.zeros(n_points)
        rcs_uncertainties = np.zeros(n_points) if model_type.lower() == 'kriging' else None

        # 扫描参数
        for j, value in enumerate(scan_values):
            # 创建新的设计参数
            new_design = base_array.copy()
            new_design[i] = value

            # 预测RCS
            pred, unc = predict_rcs(new_design, model_dict, model_type)
            rcs_predictions[j] = pred
            if rcs_uncertainties is not None:
                rcs_uncertainties[j] = unc

        # 计算参数敏感度
        sensitivity = np.max(rcs_predictions) - np.min(rcs_predictions)

        # 存储结果
        sensitivity_results[name] = {
            'scan_values': scan_values,
            'rcs_predictions': rcs_predictions,
            'rcs_uncertainties': rcs_uncertainties,
            'sensitivity': sensitivity
        }

    return sensitivity_results


def plot_sensitivity_results(sensitivity_results, top_n=None):
    setup_chinese_font()
    """
    可视化敏感度分析结果

    参数:
    sensitivity_results: 敏感度分析结果
    top_n: 显示前N个最敏感的参数
    """
    # 获取所有特征的敏感度
    features = list(sensitivity_results.keys())
    sensitivities = [sensitivity_results[f]['sensitivity'] for f in features]

    # 按敏感度排序
    sorted_indices = np.argsort(sensitivities)[::-1]  # 降序
    sorted_features = [features[i] for i in sorted_indices]
    sorted_sensitivities = [sensitivities[i] for i in sorted_indices]

    # 只保留前top_n个
    if top_n is not None and top_n < len(sorted_features):
        sorted_features = sorted_features[:top_n]
        sorted_sensitivities = sorted_sensitivities[:top_n]

    # 绘制敏感度条形图
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_sensitivities)
    plt.xlabel('设计参数')
    plt.ylabel('RCS敏感度 (dBsm)')
    plt.title('设计参数对RCS的敏感度分析')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('sensitivity_ranking.png')

    # 为前几个最敏感的参数绘制详细扫描曲线
    n_plots = min(len(sorted_features), 6)  # 最多显示6个参数
    if n_plots > 0:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for i in range(n_plots):
            feature = sorted_features[i]
            result = sensitivity_results[feature]

            ax = axes[i]
            ax.plot(result['scan_values'], result['rcs_predictions'])

            # 如果有不确定性数据，添加置信区间
            if result['rcs_uncertainties'] is not None:
                ax.fill_between(
                    result['scan_values'],
                    result['rcs_predictions'] - 1.96 * result['rcs_uncertainties'],
                    result['rcs_predictions'] + 1.96 * result['rcs_uncertainties'],
                    alpha=0.2
                )

            ax.set_xlabel(f'{feature}')
            ax.set_ylabel('RCS (dBsm)')
            ax.set_title(f'{feature} 对RCS的影响 (敏感度: {result["sensitivity"]:.4f})')
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('parameter_scan_curves.png')


def explore_design_space(model_dict, param_ranges, n_samples=1000, target_rcs=None, model_type='kriging'):
    """
    探索设计空间，寻找最优设计

    参数:
    model_dict: 模型字典
    param_ranges: 参数范围字典 {param_name: (min, max)}
    n_samples: 随机样本数量
    target_rcs: 目标RCS值，如果为None则寻找最小RCS
    model_type: 使用的模型类型

    返回:
    best_design: 最优设计参数
    """
    feature_names = model_dict['feature_names']

    # 生成随机设计
    random_designs = np.zeros((n_samples, len(feature_names)))
    for i, name in enumerate(feature_names):
        if name in param_ranges:
            min_val, max_val = param_ranges[name]
            random_designs[:, i] = np.random.uniform(min_val, max_val, n_samples)
        else:
            # 如果没有指定范围，使用默认值
            random_designs[:, i] = 0.5  # 可以根据实际情况调整

    # 预测每个设计的RCS
    predictions = []
    uncertainties = []

    for i in range(n_samples):
        design = random_designs[i]
        pred, unc = predict_rcs(design, model_dict, model_type)
        predictions.append(pred)
        uncertainties.append(unc if unc is not None else 0)

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)

    # 根据目标寻找最优设计
    if target_rcs is None:
        # 寻找最小RCS
        best_idx = np.argmin(predictions)
    else:
        # 寻找最接近目标值的设计
        best_idx = np.argmin(np.abs(predictions - target_rcs))

    best_design = {
        'parameters': {name: random_designs[best_idx, i] for i, name in enumerate(feature_names)},
        'predicted_rcs': predictions[best_idx],
        'uncertainty': uncertainties[best_idx]
    }

    print("\n最优设计:")
    print(f"预测RCS: {best_design['predicted_rcs']:.4f} dBsm")
    if best_design['uncertainty'] > 0:
        print(f"预测不确定性: {best_design['uncertainty']:.4f}")
    print("\n设计参数:")
    for name, value in best_design['parameters'].items():
        print(f"  {name}: {value:.4f}")

    return best_design


def visualize_response_surface(model_dict, param1, param2, param_ranges, fixed_params=None, resolution=20):

    """
    可视化两个参数的响应曲面

    参数:
    model_dict: 模型字典
    param1: 第一个参数名称
    param2: 第二个参数名称
    param_ranges: 参数范围字典 {param_name: (min, max)}
    fixed_params: 其他参数的固定值字典
    resolution: 网格分辨率
    """
    feature_names = model_dict['feature_names']

    # 确保参数在特征列表中
    if param1 not in feature_names or param2 not in feature_names:
        print(f"错误: 参数 {param1} 或 {param2} 不在特征列表中")
        return

    # 获取参数索引
    idx1 = feature_names.index(param1)
    idx2 = feature_names.index(param2)

    # 获取参数范围
    p1_min, p1_max = param_ranges.get(param1, (0, 1))
    p2_min, p2_max = param_ranges.get(param2, (0, 1))

    # 创建网格
    p1_values = np.linspace(p1_min, p1_max, resolution)
    p2_values = np.linspace(p2_min, p2_max, resolution)
    P1, P2 = np.meshgrid(p1_values, p2_values)

    # 创建基准设计
    if fixed_params is None:
        fixed_params = {}

    base_design = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in fixed_params:
            base_design[i] = fixed_params[name]

    # 计算RCS值
    RCS = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            design = base_design.copy()
            design[idx1] = P1[i, j]
            design[idx2] = P2[i, j]

            pred, _ = predict_rcs(design, model_dict)
            RCS[i, j] = pred

    # 可视化
    fig = plt.figure(figsize=(15, 6))

    # 等高线图
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(P1, P2, RCS, 20, cmap='viridis')
    ax1.set_xlabel(param1)
    ax1.set_ylabel(param2)
    ax1.set_title(f'RCS响应曲面 ({param1} vs {param2})')
    plt.colorbar(contour, ax=ax1, label='RCS (dBsm)')

    # 3D曲面图
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(P1, P2, RCS, cmap='viridis', alpha=0.8, edgecolor='none')
    ax2.set_xlabel(param1)
    ax2.set_ylabel(param2)
    ax2.set_zlabel('RCS (dBsm)')
    ax2.set_title(f'RCS响应曲面 ({param1} vs {param2})')

    plt.tight_layout()
    plt.savefig(f'response_surface_{param1}_{param2}.png')


def example_usage():
    """示例用法"""
    # 1. 加载模型
    model_dict = load_models('rcs_prediction_models.pkl')
    if model_dict is None:
        return

    # 2. 预测单个设计
    design_params = {
    #    'kw': 0.5,
    #    'phi': 45,
    #    'yita': 10,
    #    'lam': 10,
    #    'Lf': 0.7,
    #    'Ht': 0.4,
    #    'Nc': 2.0,
    #    'Theta': 10,
    #    'R': 15,
    #    'Beta': 8
        'kw': 0.439,
        'phi': 43.3,
        'yita': 17.4,
        'lam': 5.4,
        'Lf': 0.7,
        'Ht': 0.32,
        'Nc': 2.429,
        'Theta': 12.809,
        'R': 5,
        'Beta': 7
    }

    rcs_pred, uncertainty = predict_rcs(design_params, model_dict)
    print(f"\n设计参数RCS预测值: {rcs_pred:.4f} dBsm")
    if uncertainty is not None:
        print(f"预测不确定性: {uncertainty:.4f}")

    # 3. 比较多个设计
    designs = [
        design_params,
        {**design_params, 'kw': 0.6},
        {**design_params, 'phi': 50},
        {**design_params, 'yita': 15}
    ]
    labels = ['基准设计', '增大kw', '增大phi', '增大yita']
    comparison_df = compare_predictions(designs, model_dict, labels)

    # 4. 敏感度分析
    sensitivity_results = sensitivity_analysis(design_params, model_dict)
    plot_sensitivity_results(sensitivity_results)

    # 5. 设计空间探索
    param_ranges = {
        'kw': (0.4, 0.7),
        'phi': (30, 60),
        'yita': (0, 20),
        'lam': (0, 20),
        'Ht': (0.2, 0.5),
        'Nc': (1.0, 3.0),
        'Theta': (5, 20),
        'R': (10, 20),
        'Beta': (7, 10)
    }

    # 寻找RCS最小的设计
    best_design = explore_design_space(model_dict, param_ranges)

    # 6. 可视化响应曲面
    # 选择敏感度最高的两个参数
    sorted_params = sorted(sensitivity_results.items(),
                           key=lambda x: x[1]['sensitivity'],
                           reverse=True)
    if len(sorted_params) >= 2:
        top_param1 = sorted_params[0][0]
        top_param2 = sorted_params[1][0]

        # 使用最优设计的其他参数作为固定值
        fixed_params = best_design['parameters']
        visualize_response_surface(model_dict, top_param1, top_param2, param_ranges, fixed_params)

    print("\n示例完成，所有可视化结果已保存为图片文件。")


if __name__ == "__main__":
    example_usage()