import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.rcsetup as rcsetup

# --- 日本語表示の設定 ---
# 環境に合わせてフォントを指定してください（Windowsなら MS Gothic, Macなら Hiragino Sans など）
plt.rcParams['font.family'] = 'MS Gothic' # Windowsの場合
# plt.rcParams['font.family'] = 'AppleGothic' # Macの場合
# plt.rcParams['font.family'] = 'IPAexGothic' # Linux等でインストール済みの場合

# --- フォントサイズの一括設定 ---
plt.rcParams['font.size'] = 14          # 全体の基本サイズ
plt.rcParams['axes.titlesize'] = 18     # グラフタイトルのサイズ
plt.rcParams['axes.labelsize'] = 16     # 軸ラベルのサイズ
plt.rcParams['legend.fontsize'] = 14    # 凡例のサイズ
plt.rcParams['xtick.labelsize'] = 14    # X軸目盛りのサイズ
plt.rcParams['ytick.labelsize'] = 14    # Y軸目盛りのサイズ

# --- 設定 ---
LOG_FILE = 'LOG.csv'
OUTPUT_IMG = 'Win_vs_Lose_Final_Analysis_LargeFont.png'

def main():
    if not os.path.exists(LOG_FILE):
        print(f"{LOG_FILE} が見つかりません。")
        return

    df = pd.read_csv(LOG_FILE)
    users = df['User'].unique()
    all_rounds = sorted(df['Round'].unique())
    
    # --- データの分類と集計 ---
    winners = []
    losers = []
    box_data = []  
    line_data = [] 

    for user in users:
        u_df = df[df['User'] == user]
        prop_df = u_df[u_df['Method'] == 'Proposed'].sort_values('Round')
        rand_df = u_df[u_df['Method'] == 'Random'].sort_values('Round')
        
        if prop_df.empty or rand_df.empty:
            continue
            
        p_final_score = prop_df.iloc[-1]['Score']
        r_best_score = rand_df['Score'].max()
        
        group = 'Loser'
        if p_final_score > r_best_score:
            group = 'Winner'
            winners.append(user)
        else:
            losers.append(user)

        box_data.append({'User': user, 'Group': group, 'Method': '提案手法', 'Score': p_final_score})
        box_data.append({'User': user, 'Group': group, 'Method': 'ランダム (最高)', 'Score': r_best_score})
        
        for _, row in prop_df.iterrows():
            line_data.append({
                'User': user, 'Group': group, 'Method': '提案手法',
                'Round': row['Round'], 'Score': row['Score']
            })
            
        rand_temp = rand_df.set_index('Round')
        rand_temp['BestSoFar'] = rand_temp['Score'].cummax()
        rand_temp = rand_temp.reindex(all_rounds)
        rand_temp['BestSoFar'] = rand_temp['BestSoFar'].ffill()
        
        for r in all_rounds:
            score = rand_temp.loc[r, 'BestSoFar']
            if pd.notna(score):
                line_data.append({
                    'User': user, 'Group': group, 'Method': 'ランダム (累積最高)',
                    'Round': r, 'Score': score
                })

    df_box = pd.DataFrame(box_data)
    df_line = pd.DataFrame(line_data)

    # --- プロット作成 ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 14)) # サイズを少し拡大
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plot_configs = [
        {'title': f"全被験者群 (N={len(users)})", 'filter_group': None, 'color': 'purple', 'col': 0},
        {'title': f"提案手法適合群 (N={len(winners)})", 'filter_group': 'Winner', 'color': 'navy', 'col': 1},
        {'title': f"提案手法不適合群 (N={len(losers)})", 'filter_group': 'Loser', 'color': 'darkred', 'col': 2}
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
                          color='black', alpha=0.5, jitter=True, size=6, ax=ax_box)
            
            means = current_box.groupby('Method')['Score'].mean()
            ax_box.set_title(f"{config['title']}\n提案: {means.get('提案手法', 0):.2f} vs ランダム: {means.get('ランダム (最高)', 0):.2f}", 
                             color=main_color, fontweight='bold')
        
        ax_box.set_xlabel("")
        ax_box.set_ylabel("スコア" if col == 0 else "")
        ax_box.grid(axis='y', linestyle='--', alpha=0.7)

        # --- [下段] 折れ線グラフ ---
        ax_line = axes[1, col]
        if not current_line.empty:
            dat_prop = current_line[current_line['Method'] == '提案手法']
            dat_rand = current_line[current_line['Method'] == 'ランダム (累積最高)']

            sns.lineplot(data=dat_prop, x='Round', y='Score', units='User', estimator=None, 
                         ax=ax_line, color=main_color, alpha=0.15, linewidth=1.5)
            
            sns.lineplot(data=dat_prop, x='Round', y='Score', ax=ax_line, 
                         color=main_color, linewidth=4, errorbar=None, label='提案手法 平均',
                         marker='o', markersize=10)
            
            sns.lineplot(data=dat_rand, x='Round', y='Score', ax=ax_line, 
                         color='red', linewidth=3, linestyle='--', errorbar=None, 
                         label='ランダム 平均', marker='^', markersize=9)
            
            ax_line.set_title("ラウンド毎の平均スコア推移", color=main_color)
            
            if col == 2:
                ax_line.legend(loc='lower right')
            else:
                if ax_line.get_legend(): ax_line.get_legend().remove()
            
        ax_line.set_ylabel("スコア" if col == 0 else "")
        ax_line.set_xlabel("ラウンド")
        ax_line.set_xticks(all_rounds)
        ax_line.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"画像保存完了: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()