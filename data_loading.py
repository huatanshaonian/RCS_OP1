import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_and_match_data(rcs_file, params_file):
    """
    加载RCS数据和参数数据，并基于model_id匹配

    参数:
    rcs_file: RCS数据文件路径，第一行包含列名，第一列为model_id
    params_file: 参数数据文件路径，第一行包含参数名，行索引对应model_id

    返回:
    X: 特征矩阵
    y: RCS值
    feature_names: 特征名称
    model_ids: 模型ID
    matched_data: 匹配后的数据
    """
    print("加载数据文件...")

    try:
        # 1. 加载RCS数据（包含列名的CSV文件）
        rcs_data = pd.read_csv(rcs_file)
        print(f"RCS数据形状: {rcs_data.shape}")
        print("RCS数据前5行:")
        print(rcs_data.head())

        # 确定model_id列名
        model_id_col = rcs_data.columns[0]
        rcs_value_col = rcs_data.columns[1]
        print(f"Model ID列: '{model_id_col}'")
        print(f"RCS值列: '{rcs_value_col}'")

        # 2. 加载参数数据（第一行是参数名称）
        params_data = pd.read_csv(params_file)
        print(f"参数数据形状: {params_data.shape}")
        print("参数数据前5行:")
        print(params_data.head())

        # 获取参数名称
        feature_names = params_data.columns.tolist()
        print(f"特征列名: {feature_names}")

        # 3. 参数数据的索引是从1开始的整数，对应model_id
        # 获取RCS数据中的model_id列
        model_ids = rcs_data[model_id_col].values

        # 检查model_id是否为整数
        are_integers = all(
            isinstance(id, (int, np.integer)) or (isinstance(id, float) and id.is_integer()) for id in model_ids)
        if not are_integers:
            print("警告: model_id列包含非整数值，将尝试转换")
            # 尝试转换为整数
            try:
                model_ids = [int(id) for id in model_ids]
                print("model_id已转换为整数")
            except:
                print("无法将所有model_id转换为整数，请检查数据")
                print(f"model_id示例: {model_ids[:5]}")

        # 4. 检查model_id的范围是否在参数数据范围内
        max_param_index = len(params_data)
        valid_mask = np.array([(id > 0 and id <= max_param_index) for id in model_ids])

        if not all(valid_mask):
            invalid_count = sum(~valid_mask)
            print(f"警告: {invalid_count}个model_id超出参数数据范围(1-{max_param_index})")

            # 过滤无效model_id
            valid_rcs = rcs_data[valid_mask].reset_index(drop=True)
            valid_ids = np.array(model_ids)[valid_mask]
            print(f"过滤后剩余{len(valid_rcs)}个有效model_id")
        else:
            valid_rcs = rcs_data
            valid_ids = model_ids

        # 5. 根据model_id获取对应的参数行
        # 转换为基于0的索引
        param_indices = [id - 1 for id in valid_ids]

        # 从参数数据中提取对应行
        matched_params = params_data.iloc[param_indices].reset_index(drop=True)

        # 6. 构建结果数据框
        result = pd.DataFrame()
        result[model_id_col] = valid_rcs[model_id_col].values
        result[rcs_value_col] = valid_rcs[rcs_value_col].values

        # 添加参数列
        for col in feature_names:
            result[col] = matched_params[col].values

        print(f"匹配数据形状: {result.shape}")
        print("匹配数据前5行:")
        print(result.head())

        # 7. 提取模型数据
        X = result[feature_names].values
        y = result[rcs_value_col].values
        model_ids = result[model_id_col].values

        # 8. 保存匹配后的数据
        result.to_csv('matched_rcs_data.csv', index=False)
        print("匹配后的数据已保存至 'matched_rcs_data.csv'")

        return X, y, feature_names, model_ids, result

    except Exception as e:
        print(f"数据加载过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), [], [], None


def preprocess_data(X, y, feature_names):
    """
    为建模预处理数据

    参数:
    X: 特征矩阵
    y: 目标值
    feature_names: 特征名称

    返回:
    X_scaled: 标准化后的特征
    y: 目标值
    scaler: 标准化器
    """
    if len(X) == 0 or len(y) == 0:
        print("数据为空，无法进行预处理")
        return X, y, None

    # 转换为浮点型
    try:
        X = X.astype(float)
        y = y.astype(float)
    except Exception as e:
        print(f"数据类型转换错误: {str(e)}")

        # 尝试逐列转换
        X_float = np.zeros(X.shape)
        for i in range(X.shape[1]):
            try:
                X_float[:, i] = X[:, i].astype(float)
            except:
                print(f"无法将列 '{feature_names[i]}' 转换为浮点型")
                # 使用占位值
                X_float[:, i] = 0
        X = X_float

    # 检查缺失值
    if np.isnan(X).any() or np.isnan(y).any():
        print(f"警告: X中有{np.isnan(X).sum()}个NaN值，y中有{np.isnan(y).sum()}个NaN值")

        # 填充或删除缺失值
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        # 删除y中的NaN
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n数据已标准化")
    print(f"特征均值: {scaler.mean_}")
    print(f"特征标准差: {np.sqrt(scaler.var_)}")

    # 分析特征与目标的相关性
    if len(X) > 5:
        print("\n特征与RCS的相关性:")
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            print(f"  {name}: {corr:.4f}")

    return X_scaled, y, scaler


def main():
    """主函数"""
    # 文件路径
    rcs_file = r"..\parameter\statistics_3G.csv"
    params_file = r"..\parameter\parameters_sorted.csv"

    # 加载和匹配数据
    X, y, feature_names, model_ids, matched_data = load_and_match_data(rcs_file, params_file)

    if len(X) == 0:
        print("数据加载失败，无法继续处理")
        return

    # 数据预处理
    X_scaled, y, scaler = preprocess_data(X, y, feature_names)

    print("\n数据处理完成，可以进行模型训练")

    # 返回处理后的数据，可以传递给建模函数
    return X_scaled, y, feature_names, model_ids, scaler, matched_data


if __name__ == "__main__":
    results = main()