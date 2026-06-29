import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 設定 ---
LOG_FILE = 'LOG.csv'
OUTPUT_IMG = 'User_Score_Trend_Analysis.png'

def main():
    # データの読み込み
    df = pd.read_csv(LOG_FILE)
    
    # 提案手法(Proposed)のデータのみを抽出して分析
    # (ランダム手法はラウンド間の学習効果を測る対象ではないため除外)
    df_prop = df[df['Method'] == 'Proposed'].copy()
    
    # ピボットテーブル作成 (Index: User, Col: Round, Val: Score)
    pivoted = df_prop.pivot(index='User', columns='Round', values='Score')
    
    # 分類用辞書
    categories = {
        'Increased': [],   # 常に増加または維持
        'Decreased': [],   # 常に減少または維持
        'Fluctuated': []   # 上下変動あり
    }

    # 各ユーザーのスコア変動を判定
    for user, row in pivoted.iterrows():
        # NaNを除去してスコア配列を取得
        scores = row.dropna().values
        
        # 判定には最低2つのデータポイントが必要
        if len(scores) < 2:
            continue
            
        # 差分計算
        diffs = np.diff(scores)
        
        if all(d >= 0 for d in diffs):
            categories['Increased'].append(user)
        elif all(d <= 0 for d in diffs):
            categories['Decreased'].append(user)
        else:
            categories['Fluctuated'].append(user)

    # ---------------------------------------------------------
    # プロット作成 (1行3列)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    plot_configs = [
        {'key': 'Increased',  'title': 'Score Increased',  'color': 'forestgreen'},
        {'key': 'Decreased',  'title': 'Score Decreased',  'color': 'firebrick'},
        {'key': 'Fluctuated', 'title': 'Score Fluctuated', 'color': 'darkorange'}
    ]

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        key = config['key']
        user_list = categories[key]
        color = config['color']
        
        # 該当ユーザーのデータを抽出してプロット
        for user in user_list:
            user_data = df_prop[df_prop['User'] == user].sort_values('Round')
            sns.lineplot(data=user_data, x='Round', y='Score', ax=ax,
                         marker='o', markersize=8, linewidth=2, label=user,
                         color=sns.color_palette("tab10")[hash(user) % 10] if key == 'Fluctuated' else color,
                         alpha=0.7)
            
            # 最終ポイントにスコアを表示
            last_pt = user_data.iloc[-1]
            ax.text(last_pt['Round'] + 0.05, last_pt['Score'], f"{last_pt['Score']}", 
                    verticalalignment='center', fontsize=9, color='black')

        # グラフの体裁
        ax.set_title(f"{config['title']} (N={len(user_list)})", fontsize=14, fontweight='bold', color=color)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Round", fontsize=12)
        if i == 0:
            ax.set_ylabel("Mean Score", fontsize=12)
        else:
            ax.set_ylabel("")
            
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 凡例 (ユーザーが多い場合は見づらくなるため、右下に小さく配置)
        if len(user_list) > 0:
            ax.legend(title='User', fontsize=9, loc='lower left')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', fontsize=12, color='gray')

    plt.suptitle("Score Fluctuation Patterns by User (Proposed Method)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"画像保存完了: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()