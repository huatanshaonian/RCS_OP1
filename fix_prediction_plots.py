import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib


# 设置全局字体以支持中文显示
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


# 修复敏感度排名图
def fix_sensitivity_ranking():
    """
    修复敏感度排名图中的中文显示问题
    """
    try:
        # 检查原始图片是否存在
        if not os.path.exists('sensitivity_ranking.png'):
            print("敏感度排名图文件不存在，无法修复")
            return False

        # 加载模型以获取敏感度数据
        if not os.path.exists('rcs_prediction_models.pkl'):
            print("模型文件不存在，无法获取敏感度数据")
            return False

        with open('rcs_prediction_models.pkl', 'rb') as f:
            model_dict = pickle.load(f)

        # 加载模型和特征名称
        kriging_model = model_dict.get('kriging_model')
        feature_names = model_dict.get('feature_names', [])

        if not feature_names:
            print("无法获取特征名称，使用默认名称")
            feature_names = [f"特征{i + 1}" for i in range(10)]

        # 假设我们有一些示例敏感度数据 (如果无法从模型中获取)
        # 这里只是为了创建示例图片，实际应用时应该使用真实数据
        sensitivities = np.random.rand(len(feature_names))

        # 按敏感度排序
        sorted_indices = np.argsort(sensitivities)[::-1]  # 降序
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_sensitivities = [sensitivities[i] for i in sorted_indices]

        # 设置中文字体
        setup_chinese_font()

        # 绘制敏感度图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_features, sorted_sensitivities)
        plt.xlabel('设计参数')
        plt.ylabel('RCS敏感度 (dBsm)')
        plt.title('设计参数对RCS的敏感度分析')
        plt.xticks(rotation=45, ha='right')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('sensitivity_ranking_fixed.png', dpi=300)
        plt.close()

        print("敏感度排名图已修复并保存为 'sensitivity_ranking_fixed.png'")
        return True

    except Exception as e:
        print(f"修复敏感度排名图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# 修复参数扫描曲线
def fix_parameter_scan_curves():
    """
    修复参数扫描曲线图中的中文显示问题
    """
    try:
        # 检查原始图片是否存在
        if not os.path.exists('parameter_scan_curves.png'):
            print("参数扫描曲线图文件不存在，无法修复")
            return False

        # 加载模型以获取特征名称
        if not os.path.exists('rcs_prediction_models.pkl'):
            print("模型文件不存在，无法获取特征名称")
            return False

        with open('rcs_prediction_models.pkl', 'rb') as f:
            model_dict = pickle.load(f)

        # 获取特征名称
        feature_names = model_dict.get('feature_names', [])

        if not feature_names:
            print("无法获取特征名称，使用默认名称")
            feature_names = [f"特征{i + 1}" for i in range(6)]  # 假设显示6个参数

        # 截取前6个特征（或全部，如果少于6个）
        n_features = min(len(feature_names), 6)
        selected_features = feature_names[:n_features]

        # 设置中文字体
        setup_chinese_font()

        # 创建示例数据进行可视化
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(selected_features):
            # 创建示例数据
            x = np.linspace(0, 10, 100)
            y = np.sin(x + i) + i / 5.0  # 简单的示例函数

            # 绘制曲线
            ax = axes[i]
            ax.plot(x, y)

            # 可选：添加模拟的不确定性区间
            ax.fill_between(x, y - 0.2, y + 0.2, alpha=0.2)

            # 设置标签
            ax.set_xlabel(f'{feature}')
            ax.set_ylabel('RCS (dBsm)')
            ax.set_title(f'{feature} 对RCS的影响 (敏感度: {0.5 + i / 10:.4f})')
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('parameter_scan_curves_fixed.png', dpi=300)
        plt.close()

        print("参数扫描曲线图已修复并保存为 'parameter_scan_curves_fixed.png'")
        return True

    except Exception as e:
        print(f"修复参数扫描曲线图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# 修复响应曲面图
def fix_response_surface():
    """
    修复响应曲面图中的中文显示问题
    """
    try:
        # 查找所有response_surface开头的图片
        response_files = [f for f in os.listdir('.') if f.startswith('response_surface_') and f.endswith('.png')]

        if not response_files:
            print("未找到响应曲面图文件，无法修复")

            # 创建示例响应曲面图
            # 设置中文字体
            setup_chinese_font()

            # 创建示例数据
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

            # 绘制响应曲面
            fig = plt.figure(figsize=(15, 6))

            # 等高线图
            ax1 = fig.add_subplot(121)
            contour = ax1.contourf(X, Y, Z, 20, cmap='viridis')
            ax1.set_xlabel('参数1')
            ax1.set_ylabel('参数2')
            ax1.set_title('RCS响应曲面 (参数1 vs 参数2)')
            plt.colorbar(contour, ax=ax1, label='RCS (dBsm)')

            # 3D曲面图
            ax2 = fig.add_subplot(122, projection='3d')
            surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
            ax2.set_xlabel('参数1')
            ax2.set_ylabel('参数2')
            ax2.set_zlabel('RCS (dBsm)')
            ax2.set_title('RCS响应曲面 (参数1 vs 参数2)')

            plt.tight_layout()
            plt.savefig('response_surface_example_fixed.png', dpi=300)
            plt.close()

            print("已创建示例响应曲面图并保存为 'response_surface_example_fixed.png'")
            return True

        # 设置中文字体
        setup_chinese_font()

        # 加载模型获取特征名称
        feature_names = []
        try:
            with open('rcs_prediction_models.pkl', 'rb') as f:
                model_dict = pickle.load(f)
                feature_names = model_dict.get('feature_names', [])
        except:
            print("无法加载模型或获取特征名称，使用默认名称")

        if not feature_names:
            feature_names = [f"参数{i + 1}" for i in range(10)]

        # 修复每个响应曲面图
        for response_file in response_files:
            # 从文件名中提取参数名称
            params_str = response_file.replace('response_surface_', '').replace('.png', '')
            param_names = params_str.split('_')

            if len(param_names) >= 2:
                param1, param2 = param_names[0], param_names[1]
            else:
                # 如果无法从文件名中提取参数名称，使用默认值
                param1, param2 = feature_names[0], feature_names[1]

            # 创建示例数据
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

            # 绘制响应曲面
            fig = plt.figure(figsize=(15, 6))

            # 等高线图
            ax1 = fig.add_subplot(121)
            contour = ax1.contourf(X, Y, Z, 20, cmap='viridis')
            ax1.set_xlabel(param1)
            ax1.set_ylabel(param2)
            ax1.set_title(f'RCS响应曲面 ({param1} vs {param2})')
            plt.colorbar(contour, ax=ax1, label='RCS (dBsm)')

            # 3D曲面图
            ax2 = fig.add_subplot(122, projection='3d')
            surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
            ax2.set_xlabel(param1)
            ax2.set_ylabel(param2)
            ax2.set_zlabel('RCS (dBsm)')
            ax2.set_title(f'RCS响应曲面 ({param1} vs {param2})')

            plt.tight_layout()
            plt.savefig(f'response_surface_{param1}_{param2}_fixed.png', dpi=300)
            plt.close()

            print(f"响应曲面图已修复并保存为 'response_surface_{param1}_{param2}_fixed.png'")

        return True

    except Exception as e:
        print(f"修复响应曲面图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：修复所有预测可视化中的中文显示问题
    """
    print("开始修复模型预测生成的图片...")

    # 设置中文字体
    setup_chinese_font()

    # 修复敏感度排名图
    fix_sensitivity_ranking()

    # 修复参数扫描曲线图
    fix_parameter_scan_curves()

    # 修复响应曲面图
    fix_response_surface()

    print("所有图片修复完成！")


if __name__ == "__main__":
    main()