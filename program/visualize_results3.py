import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 設定 ---
LOG_FILE = 'LOG.csv'
OUTPUT_IMG = 'Win_vs_Lose_Final_Analysis_Both_BestSoFar.png'

def main():
    if not os.path.exists(LOG_FILE):
        print(f"{LOG_FILE} が見つかりません。")
        return

    # データの読み込み
    df = pd.read_csv(LOG_FILE)
    users = df['User'].unique()
    all_rounds = sorted(df['Round'].unique())
    
    # ---------------------------------------------------------
    # 1. データの分類と集計
    # ---------------------------------------------------------
    winners = []
    losers = []
    
    box_data = []  
    line_data = [] 

    print(f"全被験者数: {len(users)}名")
    
    for user in users:
        u_df = df[df['User'] == user]
        
        # (A) 提案手法
        prop_df = u_df[u_df['Method'] == 'Proposed'].sort_values('Round')
        
        # (B) ランダム手法
        rand_df = u_df[u_df['Method'] == 'Random'].sort_values('Round')
        
        if prop_df.empty or rand_df.empty:
            continue
            
        # 勝敗判定用（ここは変更せず、最終結果vsランダムベストで判定）
        p_final_score = prop_df.iloc[-1]['Score']
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
        
        # --- 折れ線グラフ用データ (両手法とも Best So Far を計算) ---
        
        # 1. Proposed (修正: 累積最大値 + 欠損対策)
        prop_temp = prop_df.set_index('Round')
        prop_temp['BestSoFar'] = prop_temp['Score'].cummax()
        prop_temp = prop_temp.reindex(all_rounds)
        prop_temp['BestSoFar'] = prop_temp['BestSoFar'].ffill()
        
        for r in all_rounds:
            score = prop_temp.loc[r, 'BestSoFar']
            if pd.notna(score):
                line_data.append({
                    'User': user, 'Group': group, 'Method': 'Proposed (Best so far)',
                    'Round': r, 'Score': score
                })
            
        # 2. Random (累積最大値 + 欠損対策)
        rand_temp = rand_df.set_index('Round')
        rand_temp['BestSoFar'] = rand_temp['Score'].cummax()
        rand_temp = rand_temp.reindex(all_rounds)
        rand_temp['BestSoFar'] = rand_temp['BestSoFar'].ffill()
        
        for r in all_rounds:
            score = rand_temp.loc[r, 'BestSoFar']
            if pd.notna(score):
                line_data.append({
                    'User': user, 'Group': group, 'Method': 'Random (Best so far)',
                    'Round': r, 'Score': score
                })

    print(f"  - Winner: {len(winners)}名")
    print(f"  - Loser : {len(losers)}名")

    df_box = pd.DataFrame(box_data)
    df_line = pd.DataFrame(line_data)

    # ---------------------------------------------------------
    # 2. プロット作成
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    plot_configs = [
        {'title': f"All Users (N={len(users)})", 'filter_group': None, 'color': 'purple', 'col': 0},
        {'title': f"Winners (N={len(winners)})", 'filter_group': 'Winner', 'color': 'navy', 'col': 1},
        {'title': f"Losers (N={len(losers)})", 'filter_group': 'Loser', 'color': 'darkred', 'col': 2}
    ]

    for config in plot_configs:
        col = config['col']
        group_filter = config['filter_group']
        main_color = config['color']
        
        if group_filter is None:
            current_box = df_box
            current_line = df_line
        else:
            current_box = df_box[df_box['Group'] == group_filter]
            current_line = df_line[df_line['Group'] == group_filter]

        # --- [上段] 箱ひげ図 ---
        ax_box = axes[0, col]
        if not current_box.empty:
            sns.boxplot(data=current_box, x='Method', y='Score', 
                        palette=['skyblue', 'lightcoral'], ax=ax_box, width=0.5, showfliers=False)
            sns.stripplot(data=current_box, x='Method', y='Score', 
                          color='black', alpha=0.5, jitter=True, size=5, ax=ax_box)
            
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

        # --- [下段] 折れ線グラフ ---
        ax_line = axes[1, col]
        
        if not current_line.empty:
            # 両手法とも "Best so far" タグがついたデータを抽出
            dat_prop = current_line[current_line['Method'] == 'Proposed (Best so far)']
            dat_rand = current_line[current_line['Method'] == 'Random (Best so far)']

            # 1. Proposed: 個別線 (薄い実線)
            sns.lineplot(data=dat_prop, x='Round', y='Score', units='User', estimator=None, 
                         ax=ax_line, color=main_color, alpha=0.15, linewidth=1)
            
            # 2. Proposed: 平均線 (太い実線)
            sns.lineplot(data=dat_prop, x='Round', y='Score', ax=ax_line, 
                         color=main_color, linewidth=3, errorbar=None, label='Proposed Avg (Best so far)',
                         marker='o', markersize=8)
            
            # 3. Random: 平均線のみ (太い点線)
            sns.lineplot(data=dat_rand, x='Round', y='Score', ax=ax_line, 
                         color='red', linewidth=2.5, linestyle='--', errorbar=None, 
                         label='Random Avg (Best so far)',
                         marker='^', markersize=7)
            
            ax_line.set_title("Average Score per Round (Best So Far)", fontsize=12, color=main_color)
            
            if col == 2:
                ax_line.legend(loc='lower right', fontsize=9)
            else:
                ax_line.get_legend().remove()
            
        else:
            ax_line.text(0.5, 0.5, "No Data", ha='center')

        ax_line.set_ylabel("Score" if col == 0 else "")
        ax_line.set_xlabel("Round")
        ax_line.set_xticks(sorted(current_line['Round'].unique()))
        ax_line.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"画像保存完了: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()