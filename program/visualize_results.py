import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 設定 ---
LOG_FILE = 'LOG.csv'
OUTPUT_IMG = 'Win_vs_Lose_Final_Analysis.png'

def main():
    if not os.path.exists(LOG_FILE):
        print(f"{LOG_FILE} が見つかりません。")
        return

    # データの読み込み
    df = pd.read_csv(LOG_FILE)
    users = df['User'].unique()
    
    # ---------------------------------------------------------
    # 1. データの分類と集計
    # ---------------------------------------------------------
    winners = []
    losers = []
    
    box_data = []  # 箱ひげ図用
    line_data = [] # 折れ線グラフ用

    print(f"全被験者数: {len(users)}名")
    
    for user in users:
        u_df = df[df['User'] == user]
        
        # (A) 提案手法
        prop_df = u_df[u_df['Method'] == 'Proposed'].sort_values('Round')
        if prop_df.empty: continue
            
        p_final_score = prop_df.iloc[-1]['Score']
        
        # (B) ランダム手法
        rand_df = u_df[u_df['Method'] == 'Random']
        r_best_score = -1
        if not rand_df.empty:
            r_best_score = rand_df['Score'].max()
            
        # (C) 勝敗判定
        group = 'Loser'
        if p_final_score > r_best_score:
            group = 'Winner'
            winners.append(user)
        else:
            losers.append(user)

        # 箱ひげ図用データ
        box_data.append({'User': user, 'Group': group, 'Method': 'Proposed', 'Score': p_final_score})
        box_data.append({'User': user, 'Group': group, 'Method': 'Random (Best)', 'Score': r_best_score})
        
        # 折れ線グラフ用データ
        for _, row in prop_df.iterrows():
            line_data.append({
                'User': user,
                'Group': group,
                'Round': row['Round'],
                'Score': row['Score']
            })

    print(f"  - Winner: {len(winners)}名")
    print(f"  - Loser : {len(losers)}名")

    df_box = pd.DataFrame(box_data)
    df_line = pd.DataFrame(line_data)

    # ---------------------------------------------------------
    # 2. プロット作成 (2行3列: All, Winner, Loser)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # ループ処理設定
    plot_configs = [
        {
            'title': f"All Users (N={len(users)})",
            'filter_group': None, 
            'color': 'purple',
            'col': 0
        },
        {
            'title': f"Winners (N={len(winners)})",
            'filter_group': 'Winner',
            'color': 'navy',
            'col': 1
        },
        {
            'title': f"Losers (N={len(losers)})",
            'filter_group': 'Loser',
            'color': 'darkred',
            'col': 2
        }
    ]

    # === ループで描画 ===
    for config in plot_configs:
        col = config['col']
        group_filter = config['filter_group']
        main_color = config['color']
        
        # データのフィルタリング
        if group_filter is None:
            current_box = df_box
            current_line = df_line
        else:
            current_box = df_box[df_box['Group'] == group_filter]
            current_line = df_line[df_line['Group'] == group_filter]

        # -------------------------------------------------
        # [上段] 箱ひげ図 (Score Distribution)
        # -------------------------------------------------
        ax_box = axes[0, col]
        
        if not current_box.empty:
            sns.boxplot(data=current_box, x='Method', y='Score', 
                        palette=['skyblue', 'lightcoral'], ax=ax_box, width=0.5)
            sns.stripplot(data=current_box, x='Method', y='Score', 
                          color='black', alpha=0.5, jitter=True, size=5, ax=ax_box)
            
            means = current_box.groupby('Method')['Score'].mean()
            # NaN対策（データがない場合）
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
        # [下段] 折れ線グラフ (Learning Curve) + 基準線
        # -------------------------------------------------
        ax_line = axes[1, col]
        
        if not current_line.empty:
            # 1. ランダム平均スコアの計算 (基準線用)
            rand_avg = current_box[current_box['Method'] == 'Random (Best)']['Score'].mean()

            # 2. 基準線 (Random Avg) を描画 [赤い点線]
            ax_line.axhline(y=rand_avg, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Avg Best')

            # 3. 個別の線 (Proposed) [薄い線]
            sns.lineplot(data=current_line, x='Round', y='Score', units='User', estimator=None, ax=ax_line, 
                         color=main_color, alpha=0.15, linewidth=1, marker='o', markersize=3)
            
            # 4. 平均線 (Proposed) [太い線]
            sns.lineplot(data=current_line, x='Round', y='Score', ax=ax_line, 
                         color=main_color, linewidth=3, errorbar=None, label='Proposed Avg',
                         marker='o', markersize=8, markeredgecolor='white')
            
            ax_line.set_title("Learning Curve vs Random Baseline", fontsize=12, color=main_color)
            
            # 凡例の位置調整
            ax_line.legend(loc='lower right', fontsize=9)
            
            # 基準線の数値をテキスト表示 (右端)
            ax_line.text(current_line['Round'].max(), rand_avg + 0.05, f"{rand_avg:.2f}", 
                         color='red', fontsize=10, fontweight='bold', ha='right', va='bottom')

        else:
            ax_line.text(0.5, 0.5, "No Data", ha='center')


        ax_line.set_ylabel("Score" if col == 0 else "")
        ax_line.set_xlabel("Round")
        ax_line.grid(True, linestyle='--', alpha=0.5)

    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"画像保存完了: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()