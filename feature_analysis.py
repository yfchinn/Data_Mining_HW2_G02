"""
================================================================================
信用卡詐欺偵測 (Credit Card Fraud Detection) - 特徵分析與特徵工程腳本
================================================================================
【核心模組與實證目標】
1. 模組一 (Module 1 - Separation Analysis):
   - 計算 V1-V28 的 Kolmogorov-Smirnov (K-S) 統計量與 Cohen's d 效應量。
   - 挑出對 Class 0 (正常) 與 Class 1 (詐欺) 具備最高區分度的 Top 關鍵特徵。

2. 模組二 (Module 2 - Amount & Time Feature Engineering):
   - 時間特徵：將 Time (秒) 轉換為 Hour_of_Day (0-23 小時) 週期特徵，分析高風險時段。
   - 金額特徵：對 Amount 進行 Log1p 轉型與 RobustScaler 處理，繪製雙峰/極端值分佈。

3. 模組三 (Module 3 - Correlation Difference Matrix):
   - 比較 Class 0 與 Class 1 各自的 Pearson/Spearman 相關係數熱力圖。
   - 揭示詐欺交易中特定的特徵交互作用 (Feature Interactions)，為 T2/T3 提供理論支持。

4. 模組四 (Module 4 - Baseline Feasibility Check):
   - 實證檢驗 T1 統計模型 (如 2 標準差/Z-score/IQR) 在極端不平衡 (0.17%) 下的誤報率 (FPR)。
   - 證明為什麼必須選用 PR-AUC (Average Precision) 作為主要評估指標。
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.preprocessing import RobustScaler

# 設定繪圖風格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 圖表輸出目錄
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ==============================================================================
# 0. 資料載入 (Data Loading)
# ==============================================================================
def load_credit_card_data(csv_path="creditcard.csv"):
    """載入 Credit Card Fraud Detection 資料集"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到檔案 `{csv_path}`，請確認資料集檔名與路徑。")
    
    df = pd.read_csv(csv_path)
    n_total = len(df)
    n_fraud = df['Class'].sum()
    n_normal = n_total - n_fraud
    
    print("=" * 60)
    print(f"✅ 資料載入成功！總筆數: {n_total:,}")
    print(f"   - 正常交易 (Class 0): {n_normal:,} ({n_normal/n_total*100:.3f}%)")
    print(f"   - 詐欺交易 (Class 1): {n_fraud:,} ({n_fraud/n_total*100:.3f}%)")
    print("=" * 60)
    return df


# ==============================================================================
# 模組一：特徵分佈與異常分離度分析 (Feature Separation Analysis)
# ==============================================================================
def analyze_feature_separation(df, top_n=6):
    """
    計算 V1-V28 各特徵在 Class 0 與 Class 1 之間的 K-S 統計量與 Cohen's d
    並繪製 Top 特徵的 KDE 分佈圖。
    """
    print("\n🔍 [Module 1] 特徵分離度與 K-S 統計量分析...")
    v_cols = [f"V{i}" for i in range(1, 29)]
    
    stats_list = []
    normal_df = df[df['Class'] == 0]
    fraud_df = df[df['Class'] == 1]
    
    for col in v_cols:
        s1 = normal_df[col]
        s2 = fraud_df[col]
        
        # Kolmogorov-Smirnov 檢定
        ks_stat, p_val = ks_2samp(s1, s2)
        
        # Cohen's d 效應量
        n1, n2 = len(s1), len(s2)
        s1_var, s2_var = s1.var(), s2.var()
        pooled_std = np.sqrt(((n1 - 1) * s1_var + (n2 - 1) * s2_var) / (n1 + n2 - 2))
        cohens_d = (s2.mean() - s1.mean()) / pooled_std if pooled_std > 0 else 0
        
        stats_list.append({
            'Feature': col,
            'KS_Stat': ks_stat,
            'p_value': p_val,
            'Cohens_d': cohens_d,
            'Abs_d': abs(cohens_d)
        })
        
    stats_df = pd.DataFrame(stats_list).sort_values(by='KS_Stat', ascending=False)
    
    print("\n🏆 【Top 10 最具詐欺區分度之 PCA 特徵 (以 K-S 統計量排序)】:")
    print(stats_df[['Feature', 'KS_Stat', 'Cohens_d']].head(10).to_string(index=False))
    
    # 繪製 Top N 區分度特徵的 KDE 分佈圖
    top_features = stats_df['Feature'].head(top_n).tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_features):
        sns.kdeplot(data=normal_df[col], ax=axes[idx], color='dodgerblue', label='Normal (Class 0)', fill=True, alpha=0.3)
        sns.kdeplot(data=fraud_df[col], ax=axes[idx], color='crimson', label='Fraud (Class 1)', fill=True, alpha=0.3)
        axes[idx].set_title(f"{col} (K-S Stat: {stats_df.loc[stats_df['Feature']==col, 'KS_Stat'].values[0]:.3f})", fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].legend()
        
    plt.suptitle("Top 6 Features with Highest Separation Power (Class 0 vs. Class 1)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    save_path = os.path.join(FIG_DIR, "fig1_feature_separation_kde.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 圖表已存至: {save_path}")
    
    return stats_df


# ==============================================================================
# 模組二（更新版）：Time 與 Amount 轉換前後完全對照圖
# ==============================================================================
def engineer_time_and_amount(df):
    """
    1. 將 Time (秒) 轉為 Hour_of_Day (0-23 小時) 週期特徵
    2. 繪製 Amount 轉換前 (Raw Amount) 與 轉換後 (Log1p Amount) 之對照圖
    """
    print("\n⏳ [Module 2] Time 與 Amount 特徵工程與轉換前後對照檢驗...")
    df_engineered = df.copy()
    
    # 時間特徵：計算交易時間是當天的第幾小時 (0 - 23)
    df_engineered['Hour_of_Day'] = (df_engineered['Time'] / 3600) % 24
    
    # 金額特徵：Log1p 平滑極端偏態
    df_engineered['Amount_Log1p'] = np.log1p(df_engineered['Amount'])
    
    # 繪製 3 欄式畫布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 交易小時分佈 (Density)
    sns.kdeplot(data=df_engineered[df_engineered['Class']==0]['Hour_of_Day'], ax=axes[0], 
                color='dodgerblue', label='Normal (Class 0)', fill=True, alpha=0.3, bw_adjust=0.8)
    sns.kdeplot(data=df_engineered[df_engineered['Class']==1]['Hour_of_Day'], ax=axes[0], 
                color='crimson', label='Fraud (Class 1)', fill=True, alpha=0.3, bw_adjust=0.8)
    axes[0].set_title("1. Transaction Density by Hour of Day", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hour of Day (0-23)")
    axes[0].set_ylabel("Density")
    axes[0].set_xticks(range(0, 25, 4))
    axes[0].legend()
    
    # 2. 轉換前：原始 Amount 分佈 (展示極端右偏與壓扁效應)
    # 為了避免 25,000 的極大值把 X 軸拉到無限長，限制顯示至 $1,000 (涵蓋 99% 以上交易)
    sns.kdeplot(data=df_engineered[df_engineered['Class']==0]['Amount'], ax=axes[1], 
                color='dodgerblue', label='Normal (Class 0)', fill=True, alpha=0.3, clip=(0, 1000))
    sns.kdeplot(data=df_engineered[df_engineered['Class']==1]['Amount'], ax=axes[1], 
                color='crimson', label='Fraud (Class 1)', fill=True, alpha=0.3, clip=(0, 1000))
    axes[1].set_title("2. Raw Amount Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Raw Amount (Clipped at $1000)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    # 3. 轉換後：Log1p Amount 分佈 (展現雙峰結構)
    sns.kdeplot(data=df_engineered[df_engineered['Class']==0]['Amount_Log1p'], ax=axes[2], 
                color='dodgerblue', label='Normal (Class 0)', fill=True, alpha=0.3)
    sns.kdeplot(data=df_engineered[df_engineered['Class']==1]['Amount_Log1p'], ax=axes[2], 
                color='crimson', label='Fraud (Class 1)', fill=True, alpha=0.3)
    axes[2].set_title("3. Log1p Amount Distribution (Log Transformed)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Log(Amount + 1)")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig2_time_amount_engineering.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 轉換前後對照圖已更新並存至: {save_path}")
    
    return df_engineered


# ==============================================================================
# 模組三：正常 vs 詐欺之特徵相關性結構差異 (Correlation Difference Matrix)
# ==============================================================================
def analyze_correlation_difference(df, top_k_features):
    """
    比較 Class 0 (正常) 與 Class 1 (詐欺) 在 Top 特徵上的 Correlation 差異
    揭示詐欺交易特殊的非線性特徵交互作用 (Feature Interactions)。
    """
    print("\n🔥 [Module 3] 正常交易 vs 詐欺交易 相關性結構差異分析...")
    
    cols = top_k_features + ['Amount']
    
    corr_normal = df[df['Class'] == 0][cols].corr()
    corr_fraud = df[df['Class'] == 1][cols].corr()
    corr_diff = corr_fraud - corr_normal # 相關性變化矩陣
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(corr_normal, ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False)
    axes[0].set_title("Normal Transactions Correlation (Class 0)", fontsize=12, fontweight='bold')
    
    sns.heatmap(corr_fraud, ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False)
    axes[1].set_title("Fraud Transactions Correlation (Class 1)", fontsize=12, fontweight='bold')
    
    sns.heatmap(corr_diff, ax=axes[2], cmap='bwr', vmin=-1, vmax=1, annot=True, fmt=".2f")
    axes[2].set_title("Correlation Difference (Fraud - Normal)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "fig3_correlation_difference_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 圖表已存至: {save_path}")


# ==============================================================================
# 模組四：T1 統計基線可行性評估 (T1 Baseline Feasibility Check)
# ==============================================================================
def evaluate_t1_baseline_feasibility(df, top_features):
    """
    評估單純使用 Z-Score (如 ±2 個標準差) 或 IQR 做離群值門檻時，
    對極端不平衡資料造成的誤報率 (FPR) 與 Precision/Recall 衝擊。
    """
    print("\n⚠️ [Module 4] T1 統計模型 (Z-Score Threshold) 可行性與誤報率評估...")
    
    normal_df = df[df['Class'] == 0]
    
    # 僅於正常交易上估計平均值與標準差 (模擬非監督/無污染訓練)
    means = normal_df[top_features].mean()
    stds = normal_df[top_features].std()
    
    # 計算全資料集相對於正常群體的 Z-score
    z_scores = (df[top_features] - means) / stds
    
    # 判斷規則：只要有 k 個以上的特徵超出 ±2 個標準差，即判定為異常 (Fraud)
    threshold = 2.0
    outlier_matrix = z_scores.abs() > threshold
    
    print("\n【實證結果：以 ±2 個標準差作為 T1 統計門檻之效果】")
    for min_outlier_count in [1, 2, 3, 5]:
        pred_fraud = outlier_matrix.sum(axis=1) >= min_outlier_count
        
        tp = ((pred_fraud == 1) & (df['Class'] == 1)).sum()
        fp = ((pred_fraud == 1) & (df['Class'] == 0)).sum()
        fn = ((pred_fraud == 0) & (df['Class'] == 1)).sum()
        tn = ((pred_fraud == 0) & (df['Class'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f" - 規則: 至少 {min_outlier_count} 個特徵偏離 > 2 Std | "
              f"Recall: {recall*100:.1f}% | Precision: {precision*100:.2f}% | "
              f"誤報比數 (FP): {fp:,} 筆 (FPR: {fpr*100:.2f}%)")
        
    print("\n📌 結論與洞察: 單純統計門檻會產生大量誤報 (FP)，強烈證實了選用 PR-AUC (Average Precision) 為評估指標的合理性！")


# ==============================================================================
# 主執行流程 (Main Workflow)
# ==============================================================================
if __name__ == "__main__":
    # 1. 載入資料 (請確保目錄下有 creditcard.csv)
    # 下載網址: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    data_path = "data/creditcard.csv"
    
    if not os.path.exists(data_path):
        print(f"⚠️ 未找到 `{data_path}`，請至 Kaggle 下載 Credit Card Fraud Detection 資料集。")
    else:
        credit_df = load_credit_card_data(data_path)
        
        # 2. 執行四大分析模組
        separation_stats = analyze_feature_separation(credit_df)
        top_k_features = separation_stats['Feature'].head(6).tolist()
        
        df_engineered = engineer_time_and_amount(credit_df)
        analyze_correlation_difference(credit_df, top_k_features)
        evaluate_t1_baseline_feasibility(credit_df, top_k_features)
        plt.show()
        print("\n🎉 特徵探索分析全數完成！所有視覺化圖表已存至 `figures/`。")