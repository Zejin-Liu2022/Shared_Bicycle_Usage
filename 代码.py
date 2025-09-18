# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score
import os
import sys
from contextlib import redirect_stdout

def set_chinese_font():
    # 设置matplotlib的中文字体，以确保图表能正确显示中文
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # Mac
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    print("中文字体设置成功。")

def abnormalIndex(df, output_dir):
    abnIndex = pd.Index([])
    col = df.columns
    with open(os.path.join(output_dir, '异常值索引.txt'), 'w', encoding='utf-8') as f:
        f.write("检测到以下特征的异常值索引：\n")
        for i in range(len(col)):
            if pd.api.types.is_numeric_dtype(df[col[i]]):
                s = df[col[i]]
                a = s.describe()
                high = a['75%'] + (a['75%'] - a['25%']) * 1.5
                low = a['25%'] - (a['75%'] - a['25%']) * 1.5
                abn = s[(s > high) | (s < low)]
                if not abn.empty:
                    f.write(f"  - 特征 '{col[i]}': {list(abn.index)}\n")
                abnIndex = abnIndex.union(abn.index)
    return abnIndex

def one_hot_encoding(data, column):
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True, dtype=int)], axis=1)
    data = data.drop([column], axis=1)
    return data

def train_cv(model, model_name, X, y, output_dir):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # n_jobs=-1 使用所有CPU核心并行计算
    pred = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_score = pred.mean()
    
    with open(os.path.join(output_dir, 'K折交叉验证分数.txt'), 'w', encoding='utf-8') as f:
        f.write(f'模型: {model_name}\n')
        f.write(f'交叉验证平均负均方误差: {cv_score}\n')
        f.write(f'交叉验证平均MSE: {abs(cv_score)}\n')

def main():
    # --- 1. 初始化设置 ---
    output_dir = '输出'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    set_chinese_font()

    feature_translation = {
        'instant': '记录索引', 'dteday': '日期', 'season': '季节', 'yr': '年份', 'mnth': '月份',
        'hr': '小时', 'holiday': '是否假日', 'weekday': '星期几', 'workingday': '是否工作日',
        'weathersit': '天气状况', 'temp': '温度', 'atemp': '体感温度', 'hum': '湿度',
        'windspeed': '风速', 'casual': '临时用户数', 'registered': '注册用户数', 'cnt': '总租车数'
    }

    # --- 2. 数据加载与预处理 ---
    print("开始处理：2.2.1 缺失值检查...")
    df = pd.read_csv('hour.csv')
    with open(os.path.join(output_dir, '缺失值检查.txt'), 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print("--- 各列是否存在缺失值 ---")
            print(df.isnull().any())
            print("\n--- 各列的缺失值总数 ---")
            print(df.isnull().sum())
    print("完成。结果已保存到 '输出/缺失值检查.txt'")

    print("开始处理：2.2.2 异常值处理...")
    outlier_indices = abnormalIndex(df, output_dir)
    df = df.drop(outlier_indices)
    print(f"完成。删除了 {len(outlier_indices)} 个包含异常值的行。详情已保存到 '输出/异常值索引.txt'")

    print("开始处理：2.2.3 变量转换...")
    cols_to_transform = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in cols_to_transform:
        df[col] = df[col].astype('category')
    with open(os.path.join(output_dir, '变量转换后信息.txt'), 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            df.info()
    print("完成。数据信息已保存到 '输出/变量转换后信息.txt'")

    # ###############################################################
    # ##########          新增：探索性数据分析 (EDA)           ##########
    # ###############################################################
    print("开始处理：2.3 探索性数据分析 (EDA)...")
    
    # 类别特征与租车量的关系
    categorical_features = ['season', 'hr', 'workingday', 'weathersit']
    for col in categorical_features:
        # 为小时(hr)设置更大的图像尺寸
        if col == 'hr':
            plt.figure(figsize=(16, 8))
        else:
            plt.figure(figsize=(10, 6))
        
        sns.boxplot(data=df, x=col, y='cnt')
        plt.title(f'{feature_translation[col]} 与 {feature_translation["cnt"]} 的关系', fontsize=16)
        plt.xlabel(feature_translation[col], fontsize=12)
        plt.ylabel(feature_translation["cnt"], fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature_translation[col]}与租车量关系箱型图.png'))
        plt.close()
    
    # 数值特征的分布
    numerical_features = ['temp', 'hum', 'windspeed']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    for i, col in enumerate(numerical_features):
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(f'{feature_translation[col]} 分布', fontsize=14)
        axes[i].set_ylabel(feature_translation[col], fontsize=12)
    fig.suptitle('数值型特征分布箱型图', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    plt.savefig(os.path.join(output_dir, '数值特征分布箱型图.png'))
    plt.close()
    print("完成。EDA相关的箱型图已保存。")


    print("开始处理：2.4.2 相关性分析...")
    corr_df = df.copy()
    for col in cols_to_transform:
        corr_df[col] = corr_df[col].cat.codes
    if 'dteday' in corr_df.columns:
        corr_df = corr_df.drop('dteday', axis=1)
    corr = corr_df.corr()
    plt.figure(figsize=(24, 18))
    translated_corr = corr.rename(columns=feature_translation, index=feature_translation)
    sns.heatmap(translated_corr, annot=True, annot_kws={'size': 12}, fmt='.2f', cmap='coolwarm')
    plt.title('特征相关性热力图', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '相关性分析热力图.png'))
    plt.close()
    with open(os.path.join(output_dir, '各特征与cnt相关性.txt'), 'w', encoding='utf-8') as f:
        f.write(corr['cnt'].sort_values(ascending=False).to_string())
    print("完成。热力图和相关性排序已保存。")

    print("开始处理：2.5.1 One-Hot 编码...")
    df_oh = df.copy()
    cols_to_encode = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in cols_to_encode:
        df_oh = one_hot_encoding(df_oh, col)
    with open(os.path.join(output_dir, '独热编码后数据头部.txt'), 'w', encoding='utf-8') as f:
        f.write(df_oh.head().to_string())
    print("完成。编码后数据头部已保存。")

    # --- 3. 模型构建与评估 ---
    print("开始处理：3.1 训练/测试集划分...")
    X = df_oh.drop(columns=['dteday', 'atemp', 'windspeed', 'casual', 'registered', 'cnt'], axis=1)
    y = df_oh['cnt']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    with open(os.path.join(output_dir, '训练集维度.txt'), 'w', encoding='utf-8') as f:
        f.write(f"合并后的训练集维度: {pd.concat([x_train, y_train], axis=1).shape}")
    print("完成。训练集维度信息已保存。")

    print("开始处理：3.2 K-折交叉验证...")
    train_cv(RandomForestRegressor(random_state=42), "RandomForestRegressor", X, y, output_dir)
    print("完成。交叉验证分数已保存。")

    print("开始处理：3.3.2 随机森林模型训练...")
    model_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_rf.fit(x_train, y_train)
    y_pred_rf = model_rf.predict(x_test)
    print("完成。随机森林模型已训练。")

    print("开始处理：3.4 线性回归模型训练...")
    scaler_x = StandardScaler()
    x_train_std = scaler_x.fit_transform(x_train)
    sgd = SGDRegressor(random_state=42)
    sgd.fit(x_train_std, y_train)
    with open(os.path.join(output_dir, '线性回归模型系数.txt'), 'w', encoding='utf-8') as f:
        f.write(f'模型系数 (coef_): {sgd.coef_}\n')
        f.write(f'模型截距 (intercept_): {sgd.intercept_}\n')
    print("完成。线性回归模型系数已保存。")

    print("开始处理：3.5 模型评估可视化...")
    error = y_test - y_pred_rf
    fig, ax = plt.subplots(figsize=(10, 6))
    # 为散点图和水平线添加label，以便创建图例
    ax.scatter(y_test, error, alpha=0.5, label='残差 (实际值 - 预测值)') # <--- 修改
    ax.axhline(lw=2, color='black', linestyle='--', label='零误差线') # <--- 修改
    ax.set_xlabel('观测值 (y_test)', fontsize=14)
    ax.set_ylabel('残差 (y_test - y_pred)', fontsize=14)
    ax.set_title('随机森林模型残差图', fontsize=16)
    ax.legend() # <--- 新增：显示图例
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '随机森林模型残差图.png'))
    plt.close()
    print("完成。模型残差图已保存。")

    print("开始处理：3.6.1 Min-Max 标准化...")
    data_train = pd.concat([x_train, y_train], axis=1)
    numericalFeatureNames_to_scale = ['temp', 'hum']
    scaler_mm = MinMaxScaler()
    data_train[numericalFeatureNames_to_scale] = scaler_mm.fit_transform(data_train[numericalFeatureNames_to_scale])
    print("完成。训练集中的数值特征已进行Min-Max标准化。")

    print("开始处理：3.6.2 模型再评估...")
    train_x_scaled = data_train.drop(["cnt"], axis=1)
    train_y_scaled = data_train["cnt"]
    data_test = pd.concat([x_test, y_test], axis=1)
    data_test[numericalFeatureNames_to_scale] = scaler_mm.transform(data_test[numericalFeatureNames_to_scale])
    test_x_scaled = data_test.drop(["cnt"], axis=1)
    test_y_scaled = data_test["cnt"]
    model_reeval = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_reeval.fit(train_x_scaled, train_y_scaled)
    pred_y_reeval = model_reeval.predict(test_x_scaled)
    mse = mean_squared_error(test_y_scaled, pred_y_reeval)
    evs = explained_variance_score(test_y_scaled, pred_y_reeval)
    rmse = np.sqrt(mse)
    with open(os.path.join(output_dir, '再评估的随机森林模型性能.txt'), 'w', encoding='utf-8') as f:
        f.write("随机森林回归器 (对部分特征标准化后)的性能指标：\n")
        f.write(f"  - 均方误差 (MSE): {mse}\n")
        f.write(f"  - 可释方差分数: {evs}\n")
        f.write(f"  - 均方根误差 (RMSE): {rmse}\n")
    print("完成。模型再评估的性能指标已保存。")
    print("\n--- 所有任务已完成！请检查 '输出' 文件夹中的结果。 ---")

if __name__ == "__main__":
    main()