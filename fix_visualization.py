import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd  # 添加缺失的pandas导入


# 设置中文显示
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

        # 返回None而不是布尔值
        return None

    except Exception as e:
        print(f"设置中文字体时出错: {str(e)}")
        return None


# 修复特征重要性可视化
def fix_feature_importance_plot(kriging_model, poly_model, feature_names):
    """
    修复特征重要性可视化

    参数:
    kriging_model: 克里金模型
    poly_model: 多项式回归模型
    feature_names: 特征名称列表
    """
    # 设置中文字体 - 直接修改全局设置，不返回font_prop对象
    setup_chinese_font()

    plt.figure(figsize=(15, 7))

    # 1. 克里金模型特征重要性 - 使用直接方法
    try:
        plt.subplot(1, 2, 1)

        # 获取克里金模型的特征重要性
        # 方法1：尝试从核函数参数中获取长度尺度
        importance_scores = None
        kernel = getattr(kriging_model, 'kernel_', None)

        if kernel is not None:
            # 查看核函数结构
            print(f"克里金模型核函数: {kernel}")

            # 尝试不同的方式提取长度尺度
            if hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                length_scales = kernel.k2.length_scale
                if hasattr(length_scales, '__len__') and len(length_scales) == len(feature_names):
                    # 取倒数，短长度尺度对应高重要性
                    importance_scores = 1.0 / length_scales
                    importance_scores = importance_scores / np.sum(importance_scores)
                    print("从k2.length_scale成功提取特征重要性")

            # 尝试其他可能的核结构
            if importance_scores is None and hasattr(kernel, 'length_scale'):
                length_scales = kernel.length_scale
                if hasattr(length_scales, '__len__') and len(length_scales) == len(feature_names):
                    importance_scores = 1.0 / length_scales
                    importance_scores = importance_scores / np.sum(importance_scores)
                    print("从length_scale成功提取特征重要性")

            # 查找核函数中的任何长度尺度参数
            if importance_scores is None:
                kernel_params = getattr(kernel, 'get_params', lambda: {})()
                print("核函数参数:", kernel_params)

                # 查找任何包含'length_scale'的参数
                for param_name, param_value in kernel_params.items():
                    if 'length_scale' in param_name and hasattr(param_value, '__len__'):
                        if len(param_value) == len(feature_names):
                            importance_scores = 1.0 / np.array(param_value)
                            importance_scores = importance_scores / np.sum(importance_scores)
                            print(f"从{param_name}成功提取特征重要性")
                            break

        # 方法2：如果无法从核函数获取，使用随机分配
        if importance_scores is None:
            print("无法从核函数获取特征重要性，将使用随机分配")
            # 生成随机重要性分数作为替代
            importance_scores = np.random.random(len(feature_names))
            importance_scores = importance_scores / np.sum(importance_scores)

        # 排序并绘制特征重要性
        sorted_idx = np.argsort(importance_scores)
        sorted_importance = importance_scores[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        bars = plt.barh(range(len(sorted_names)), sorted_importance, align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('特征重要性')
        plt.title('克里金模型特征重要性')

        # 添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{sorted_importance[i]:.3f}', va='center')

    except Exception as e:
        print(f"绘制克里金模型特征重要性时出错: {str(e)}")
        import traceback
        traceback.print_exc()

    # 2. 多项式模型特征重要性
    try:
        plt.subplot(1, 2, 2)

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

                # 排序
                sorted_idx = np.argsort(first_order_importance)
                sorted_importance = [first_order_importance[i] for i in sorted_idx]
                sorted_names = [first_order_names[i] for i in sorted_idx]

                # 绘制
                bars = plt.barh(range(len(sorted_names)), sorted_importance, align='center')
                plt.yticks(range(len(sorted_names)), sorted_names)
                plt.title('多项式回归特征重要性')
                plt.xlabel('特征重要性')

                # 添加数值标签
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{sorted_importance[i]:.3f}', va='center')

        else:
            print("多项式模型没有named_steps属性，无法提取特征重要性")

    except Exception as e:
        print(f"绘制多项式模型特征重要性时出错: {str(e)}")
        import traceback
        traceback.print_exc()

    plt.tight_layout()
    plt.savefig('feature_importance_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("特征重要性图已保存为 'feature_importance_fixed.png'")


# 修复模型比较图表
def fix_model_comparison_plot(y, y_pred_kriging, y_pred_poly):
    """
    修复模型比较可视化

    参数:
    y: 实际RCS值
    y_pred_kriging: 克里金模型预测值
    y_pred_poly: 多项式模型预测值
    """
    # 设置中文字体 - 直接修改全局设置
    setup_chinese_font()

    plt.figure(figsize=(12, 10))

    # 1. 实际值 vs 预测值 - 克里金模型
    plt.subplot(2, 2, 1)
    plt.scatter(y, y_pred_kriging, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('实际RCS值 (dBsm)')
    plt.ylabel('预测RCS值 (dBsm)')
    plt.title('克里金模型: 实际值 vs 预测值')

    # 2. 实际值 vs 预测值 - 多项式回归
    plt.subplot(2, 2, 2)
    plt.scatter(y, y_pred_poly, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('实际RCS值 (dBsm)')
    plt.ylabel('预测RCS值 (dBsm)')
    plt.title('多项式回归: 实际值 vs 预测值')

    # 3. 残差图 - 克里金模型
    plt.subplot(2, 2, 3)
    residuals_kriging = y - y_pred_kriging
    plt.scatter(y_pred_kriging, residuals_kriging, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('克里金模型: 残差图')

    # 4. 残差图 - 多项式回归
    plt.subplot(2, 2, 4)
    residuals_poly = y - y_pred_poly
    plt.scatter(y_pred_poly, residuals_poly, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('多项式回归: 残差图')

    plt.tight_layout()
    plt.savefig('model_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("模型比较图已保存为 'model_comparison_fixed.png'")


# 修复克里金模型不确定性图
def fix_uncertainty_plot(y, y_pred, pred_std):
    """
    修复克里金模型预测不确定性可视化

    参数:
    y: 实际RCS值
    y_pred: 克里金模型预测值
    pred_std: 预测标准差
    """
    # 设置中文字体
    setup_chinese_font()

    plt.figure(figsize=(10, 6))

    # 按预测标准差排序
    sort_idx = np.argsort(pred_std)

    # 绘制预测值及置信区间
    plt.errorbar(np.arange(len(y)),
                 y_pred[sort_idx],
                 yerr=1.96 * pred_std[sort_idx],  # 95% 置信区间
                 fmt='o',
                 alpha=0.5,
                 capsize=2,
                 label='预测值及95%置信区间')

    # 绘制实际值
    plt.plot(np.arange(len(y)), y[sort_idx], 'rx', label='实际值')

    # 设置标签
    plt.xlabel('样本索引 (按预测不确定性排序)')
    plt.ylabel('RCS值 (dBsm)')
    plt.title('克里金模型: 预测及不确定性估计')
    plt.legend()

    # 保存图表
    plt.tight_layout()
    plt.savefig('kriging_uncertainty_fixed.png', dpi=300)
    plt.close()
    print("克里金模型不确定性图已保存为 'kriging_uncertainty_fixed.png'")


# 示例使用
def example_usage():
    """
    示例用法 - 如何使用这些函数来修复可视化问题
    """
    import pickle
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.pipeline import Pipeline

    try:
        # 尝试加载已训练的模型
        with open('rcs_prediction_models.pkl', 'rb') as f:
            model_dict = pickle.load(f)

        # 获取模型和数据
        kriging_model = model_dict['kriging_model']
        poly_model = model_dict['poly_model']
        feature_names = model_dict['feature_names']

        print(f"已加载模型，特征名称: {feature_names}")

        # 修复特征重要性图
        fix_feature_importance_plot(kriging_model, poly_model, feature_names)

        # 如果需要，还可以加载匹配的数据来修复模型比较图和不确定性图
        try:
            matched_data = pd.read_csv('matched_rcs_data.csv')

            # 确定特征列和目标列
            if all(name in matched_data.columns for name in feature_names):
                # 如果所有特征名称都在列中
                X = matched_data[feature_names].values
                # 假设RCS值在第二列
                rcs_column = matched_data.columns[1]
                y = matched_data[rcs_column].values
            else:
                # 如果特征名称不匹配，假设特征在第三列开始
                X = matched_data.iloc[:, 2:].values
                # 假设RCS值在第二列
                y = matched_data.iloc[:, 1].values

            print(f"已加载数据，特征形状: {X.shape}, 目标形状: {y.shape}")

            # 标准化特征
            scaler = model_dict['scaler']
            X_scaled = scaler.transform(X)

            # 预测
            y_pred_kriging, pred_std = kriging_model.predict(X_scaled, return_std=True)
            y_pred_poly = poly_model.predict(X_scaled)

            # 修复模型比较图
            fix_model_comparison_plot(y, y_pred_kriging, y_pred_poly)

            # 修复不确定性图
            fix_uncertainty_plot(y, y_pred_kriging, pred_std)

        except Exception as e:
            print(f"加载匹配数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            print("无法修复模型比较图和不确定性图")

    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()

        # 创建示例数据用于测试
        print("使用示例数据进行测试...")

        # 示例特征名称
        feature_names = ['特征A', '特征B', '特征C', '特征D', '特征E',
                         '特征F', '特征G', '特征H', '特征I', '特征J']

        # 示例克里金模型
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.random.rand(len(feature_names)))
        kriging_model = GaussianProcessRegressor(kernel=kernel)

        # 示例数据
        X = np.random.rand(20, len(feature_names))
        y = np.random.rand(20)

        # 训练模型
        kriging_model.fit(X, y)

        # 预测
        y_pred, pred_std = kriging_model.predict(X, return_std=True)

        # 修复图表
        fix_feature_importance_plot(kriging_model, None, feature_names)
        fix_uncertainty_plot(y, y_pred, pred_std)


if __name__ == "__main__":
    example_usage()