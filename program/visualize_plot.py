import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 設定 ---
LOG_FILE = 'LOG.csv'
OUTPUT_IMG = 'Win_vs_Lose_Round_Comparison.png'

def main():
    if not os.path.exists(LOG_FILE):
        print(f"{LOG_FILE} が見つかりません。")
        return

    # データの読み込み
    df = pd.read_csv(LOG_FILE)
    users = df['User'].unique()
    
    # ---------------------------------------------------------
    # 1. ユーザーの勝敗分類 (Winner / Loser)
    # ---------------------------------------------------------
    winners = []
    losers = []
    
    # 箱ひげ図用のデータリスト (集計用)
    box_data = []

    print(f"全被験者数: {len(users)}名")
    
    for user in users:
        u_df = df[df['User'] == user]
        
        # (A) 提案手法 (Proposed)
        prop_df = u_df[u_df['Method'] == 'Proposed'].sort_values('Round')
        if prop_df.empty: continue
        p_final_score = prop_df.iloc[-1]['Score']
        
        # (B) ランダム手法 (Random)
        rand_df = u_df[u_df['Method'] == 'Random']
        r_best_score = -1
        if not rand_df.empty:
            r_best_score = rand_df['Score'].max()
            
        # (C) 勝敗判定 (提案の最終 > ランダムのベスト なら勝ち)
        group = 'Loser'
        if p_final_score > r_best_score:
            group = 'Winner'
            winners.append(user)
        else:
            losers.append(user)

        # 箱ひげ図用データの蓄積
        box_data.append({'User': user, 'Group': group, 'Method': 'Proposed', 'Score': p_final_score})
        box_data.append({'User': user, 'Group': group, 'Method': 'Random (Best)', 'Score': r_best_score})

    print(f"  - Winner: {len(winners)}名")
    print(f"  - Loser : {len(losers)}名")

    df_box = pd.DataFrame(box_data)

    # ---------------------------------------------------------
    # 2. プロット作成 (2行3列)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 余白調整
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # ループ設定
    plot_configs = [
        {'title': f"All Users (N={len(users)})", 'filter_users': list(users), 'color': 'purple', 'col': 0},
        {'title': f"Winners (N={len(winners)})", 'filter_users': winners,    'color': 'navy',   'col': 1},
        {'title': f"Losers (N={len(losers)})",   'filter_users': losers,     'color': 'darkred','col': 2}
    ]

    for config in plot_configs:
        col = config['col']
        target_users = config['filter_users']
        main_color = config['color']
        
        # --- 対象データの抽出 ---
        # 箱ひげ図用
        current_box = df_box[df_box['User'].isin(target_users)]
        
        # 折れ線グラフ用（全履歴データから抽出）
        current_raw = df[df['User'].isin(target_users)]
        
        # -------------------------------------------------
        # [上段] 箱ひげ図 (Score Distribution)
        # -------------------------------------------------
        ax_box = axes[0, col]
        
        if not current_box.empty:
            sns.boxplot(data=current_box, x='Method', y='Score', 
                        palette=['skyblue', 'lightcoral'], ax=ax_box, width=0.5)
            sns.stripplot(data=current_box, x='Method', y='Score', 
                          color='black', alpha=0.5, jitter=True, size=5, ax=ax_box)
            
            # 平均スコア
            means = current_box.groupby('Method')['Score'].mean()
            p_mean = means.get('Proposed', 0)
            r_mean = means.get('Random (Best)', 0)
            
            ax_box.set_title(f"{config['title']}\nProp: {p_mean:.2f} vs Rand: {r_mean:.2f}", 
                             fontsize=14, color=main_color, fontweight='bold')
        else:
            ax_box.text(0.5, 0.5, "No Data", ha='center')
            ax_box.set_title(config['title'], fontsize=14, color=main_color)

        ax_box.grid(axis='y', linestyle='--', alpha=0.7)
        ax_box.set_xlabel("")

        # -------------------------------------------------
        # [下段] 折れ線グラフ (Proposed Avg vs Random Avg per Round)
        # -------------------------------------------------
        ax_line = axes[1, col]
        
        if not current_raw.empty:
            # データを手法別に分ける
            prop_data = current_raw[current_raw['Method'] == 'Proposed']
            rand_data = current_raw[current_raw['Method'] == 'Random']

            # 1. ランダム手法のラウンド平均 (赤点線)
            # ※ランダムにもRound列があると仮定して集計
            if not rand_data.empty:
                sns.lineplot(data=rand_data, x='Round', y='Score', ax=ax_line,
                             color='red', linestyle='--', linewidth=2, 
                             marker='^', markersize=8, errorbar=None, label='Random Avg')

            # 2. 提案手法のラウンド平均 (青実線)
            if not prop_data.empty:
                # 個別線 (薄く)
                sns.lineplot(data=prop_data, x='Round', y='Score', units='User', estimator=None, ax=ax_line, 
                             color=main_color, alpha=0.15, linewidth=1, marker='o', markersize=3)
                # 平均線 (太く)
                sns.lineplot(data=prop_data, x='Round', y='Score', ax=ax_line, 
                             color=main_color, linewidth=3, errorbar=None, label='Proposed Avg',
                             marker='o', markersize=9, markeredgecolor='white')
            
            # X軸のメモリを 1, 2, 3 に固定
            max_round = current_raw['Round'].max()
            if np.isnan(max_round): max_round = 3
            ax_line.set_xticks(np.arange(1, int(max_round) + 1))
            
            ax_line.set_title("Average Score per Round", fontsize=12, color=main_color)
            ax_line.legend(loc='lower right', fontsize=9)
            
        else:
            ax_line.text(0.5, 0.5, "No Data", ha='center')

        ax_line.set_ylabel("Score" if col == 0 else "")
        ax_line.set_xlabel("Round")
        ax_line.grid(True, linestyle='--', alpha=0.5)

    # 全体タイトルは削除しました
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"画像保存完了: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()