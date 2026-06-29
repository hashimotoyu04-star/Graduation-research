import pysofaconventions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import pandas as pd
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from adjustText import adjust_text

#========================================================================
#========================================================================
#========================================================================

matplotlib.rcParams['font.family'] = 'Yu Gothic'
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

HRTF_DATA_DIR = 'ari'
HRTF_FILE_LIST = glob.glob(os.path.join(HRTF_DATA_DIR, '*.sofa'))

LATENT_DIM = 20
AE_EPOCHS = 50
BATCH_SIZE = 64
MODEL_PATH = 'CNN_AE_model(2,7,33).pth'

N_CLUSTERS = 10

if not HRTF_FILE_LIST:
    print(f"Error: {HRTF_DATA_DIR}ディレクトリにsofaファイルが見つかりません")
    exit()

#===================================================================================
#===================================================================================
#==================================================================================

def get_sorted_idx(sofa):
    SP = sofa.getVariableValue("SourcePosition")
    azi = np.mod(SP[:,0], 360)
    ele = SP[:,1]

    df = pd.DataFrame({
        'az': azi,
        'el': ele,
        'idx': range(len(azi))
    })

    df = df.round(3)
    df_sorted = df.sort_values(by=['az','el'])
    return df_sorted['idx'].values

print("座標順序の整合性チェック")

base_coords = None
base_filename = ""
mismatch_count = 0

for sofafile in HRTF_FILE_LIST:
    try:
        sofa = pysofaconventions.SOFAFile(sofafile, 'r')
        current_dir = sofa.getDataIR().shape[0]
        sorted_idx = get_sorted_idx(sofa)

        pos = sofa.getVariableValue("SourcePosition")
        pos[:,0] = np.mod(pos[:,0], 360)
        pos = np.round(pos, 3)
        sorted_pos = pos[sorted_idx, :2]

        if base_coords is None:
            base_coords = sorted_pos
            base_filename = os.path.basename(sofafile)
            print(f"基準HRTF 設定: {base_filename} (方向数:{current_dir})")
        else:
            if not np.array_equal(base_coords, sorted_pos):
                print(f"不一致検出: {os.path.basename(sofafile)}")
                diff = np.abs(base_coords - sorted_pos)
                max_diff = np.max(diff)
                print(f"最大ずれ: {max_diff}度")
                mismatch_count += 1
            else:
                pass
    except Exception as e:
        print(f"読み込みエラー: {sofafile} - {e}")

if mismatch_count == 0:
    print("\n✅ 全てのファイルの座標順序が完全に一致しました！")
    print("   自信を持って Flatten 結合を行えます。")
else:
    print(f"\n⚠️ 合計 {mismatch_count} ファイルで座標順序の不一致が見つかりました。")
    print("   これらをデータセットから除外するか、ソートロジックを見直す必要があります。")

def compute_spectrogram(IR, SR, nperseg=64, noverlap=32):
    all_specs = []
    num_dir = IR.shape[0] # 方向数

    for i in range(num_dir):
        irL = IR[i,0,:]
        irR = IR[i,1,:]

        f_L, t_L, Z_L = stft(irL, SR, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='spectrum')
        f_R, t_R, Z_R = stft(irR, SR, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='spectrum')


        spec_L_db = 20 * np.log10(np.abs(Z_L).T + 1e-12)
        spec_R_db = 20 * np.log10(np.abs(Z_R).T + 1e-12)

        spec_combined = np.stack([spec_L_db, spec_R_db], axis=-1)
        all_specs.append(spec_combined)

    return np.array(all_specs), f_L, t_L

print("データのロードとスペクトログラムの生成")
all_specs_for_training = []
file_dir_counts = []

for sofafile in HRTF_FILE_LIST:
    try:
        sofa = pysofaconventions.SOFAFile(sofafile, 'r')
        IR = sofa.getDataIR().data
        SR = int(sofa.getVariableValue("Data.SamplingRate"))

        specs, freqs, times = compute_spectrogram(IR,SR)
        all_specs_for_training.append(specs)
        file_dir_counts.append(IR.shape[0])

        print(f"HRTF: {os.path.basename(sofafile)}, 方向数: {IR.shape[0]}")
    except Exception as e:
        print(f"Skip {sofafile}: {e}")

full_spec_data = np.concatenate(all_specs_for_training, axis=0) # 方向軸で結合

spec_mean = full_spec_data.mean()
spec_std = full_spec_data.std()
full_spec_data_norm = (full_spec_data - spec_mean) / (spec_std + 1e-8)

data_tensor = torch.tensor(full_spec_data_norm, dtype=torch.float32).permute(0,3,1,2).to(device)
N, C, H, W = data_tensor.shape
print(f"Data Shape: {data_tensor.shape}")



#============================================================================
#============================================================================
#============================================================================

class CNN_AE(nn.Module):
    def __init__(self, latent_dim, C, H, W):
        super(CNN_AE, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1,C,H,W)
            dummy_output = self.encoder_conv(dummy_input)

            self.H_out = dummy_output.shape[2]
            self.W_out = dummy_output.shape[3]
            self.ch_out = dummy_output.shape[1]

        self.F_dim = self.ch_out * self.H_out * self.W_out

        self.encoder_Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder_Linear = nn.Sequential(
            nn.Linear(latent_dim, self.F_dim),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(0,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, kernel_size=(3,3), stride=(1,2), padding=(1,1), output_padding=(0,0)),
            nn.Identity()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_Linear(x)
        latent = x

        x = self.decoder_Linear(latent)
        x = x.view(-1, self.ch_out, self.H_out, self.W_out)
        reconstructed = self.decoder_conv(x)

        return latent, reconstructed
    
model = CNN_AE(LATENT_DIM, C, H, W).to(device)

#=============================================================================
#=============================================================================
#=============================================================================

if os.path.exists(MODEL_PATH):
    print(f"学習済みモデルをロード: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("新規学習開始")
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader =DataLoader(dataset, BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(AE_EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            opt.zero_grad()
            latent, outputs = model(inputs)
            loss = crit(outputs, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{AE_EPOCHS}], Loss: {total_loss/len(dataloader):.6f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("モデル保存完了")

#================================================================================
#================================================================================
#================================================================================

print("特徴量の抽出と結合")
model.eval()
dataset_eval = TensorDataset(data_tensor)
dataloader_eval = DataLoader(dataset_eval, BATCH_SIZE, shuffle=False)

all_latent_vector = []

with torch.no_grad():
    for batch in dataloader_eval:
        inputs = batch[0].to(device)
        latent, _ = model(inputs)
        all_latent_vector.append(latent.cpu().numpy())

all_latent_vector = np.concatenate(all_latent_vector, axis=0)

file_latent_feat = [] # ファイルごとの特徴量リスト
valid_file_idx = []

start_idx = 0
base_num_dir = file_dir_counts[0]

print(f"基準方向数: {base_num_dir} (この方向数以外のファイルは除外)")

for i, num_dir in enumerate(file_dir_counts):
    end_idx = start_idx + num_dir
    file_vec = all_latent_vector[start_idx:end_idx]

    if num_dir == base_num_dir:
        flattend_vec = file_vec.flatten() # (num_dir, 20) -> (num_dir*20, )
        file_latent_feat.append(flattend_vec)
        valid_file_idx.append(i)
    else:
        print(f"除外: {os.path.base(HRTF_FILE_LIST[i])} (方向数 {num_dir}が不一致)")

    start_idx = end_idx

final_latent_feat = np.array(file_latent_feat) # (ファイル数, 方向数*20)
print(f"クラスタリング用特徴量形状: {final_latent_feat.shape}")

#====================================================================================
#====================================================================================
#====================================================================================

print(f"K-Medoids クラスタリングと代表点抽出 (K={N_CLUSTERS})")

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(final_latent_feat)

kmedoids = KMedoids(n_clusters=N_CLUSTERS, metric='euclidean', method='pam', random_state=42)
clusters = kmedoids.fit_predict(scaled_feat)

medoid_idx = kmedoids.medoid_indices_ # 代表点のインデックスリスト

print("結果: 各クラスタの代表HRTFとファイル数")
cluster_info = {}

represetative_file_list = []

for cid in range(N_CLUSTERS):
    count = np.sum(clusters == cid) # 

    medoid_HRTF = medoid_idx[cid] # 
    orig_idx = valid_file_idx[medoid_HRTF] # 
    medoid_fname = os.path.basename(HRTF_FILE_LIST[orig_idx])

    represetative_file_list.append(medoid_fname)

    cluster_info[cid] = {'count': count, 'medoid': medoid_fname, 'idx': np.where(clusters == cid)[0]}

    print(f"Cluster {cid}: {count} files")
    print(f"  -> 代表HRTF: {medoid_fname}")

print(f"代表HRTFリスト\n{represetative_file_list}")


#==========================================================================
#==========================================================================
#==========================================================================

print("可視化")
plt.figure(figsize=(14,10))

pca = PCA(n_components=2)
feat_2d = pca.fit_transform(scaled_feat)

palette = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8']

# 1. 全データの散布図（背景）
for cid in range(N_CLUSTERS):
    mask = (clusters == cid)
    xy = feat_2d[mask]
    color = palette[cid]
    marker = markers[cid % len(markers)]

    plt.scatter(
        xy[:, 0], xy[:, 1],
        c=[color],
        marker=marker,
        label=f"Cluster {cid}",
        alpha=0.9,           # くっきり表示
        s=50,
        edgecolors='black',
        linewidths=0.5,
        antialiased=False    # ドット感を出してくっきり
    )

# 2. 代表点の描画とテキスト準備
medoid_points = feat_2d[medoid_idx]
texts = []  # adjustText用にテキストオブジェクトを溜めるリスト

for cid in range(N_CLUSTERS):
    mx, my = medoid_points[cid]
    color = palette[cid]
    fname = cluster_info[cid]['medoid']

    # --- 代表点の装飾（ハロー効果 + 星） ---
    # 白い背景丸（見やすくするため）

    
    # 星型プロット
    plt.scatter(
        mx, my,
        c=[color], s=300, marker='*',
        edgecolors='black', linewidths=1.0,
        zorder=10
    )

    # --- テキストオブジェクトの作成 ---
    # ここではまだ表示位置を確定せず、オブジェクトを作ってリストに入れるだけ
    t = plt.text(
        mx, my,
        fname,
        fontsize=9,
        fontweight='bold',
        color='black',
        ha='center', va='center', # 初期位置は中心合わせ
        # 背景ボックスのデザイン
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec=color,   # 枠線をクラスタ色に合わせる
            lw=2,
            alpha=0.4
        ),
        zorder=11
    )
    texts.append(t)

# 3. adjustTextを実行（ここで自動調整）
# 矢印を出しつつ、全データ点(feat_2d)を避けるように配置
adjust_text(
    texts,
    x=feat_2d[:, 0],      # 避けるべき全データのX座標
    y=feat_2d[:, 1],      # 避けるべき全データのY座標
    arrowprops=dict(
        arrowstyle='-|>', 
        color='black', 
        lw=1.5,
        connectionstyle="arc3,rad=0.2" # 少しカーブした線
    ),
    expand_points=(3.0, 3.0), # 点から少し強めに離す設定
    force_text=2.0,     # テキスト同士の反発力
    force_points=3.0
)

# --- 仕上げ ---
plt.title(f"HRTF Clustering Result (K={N_CLUSTERS})", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=14)
plt.ylabel("PCA Component 2", fontsize=14)
plt.legend()
plt.grid()
# adjustTextを使うときは tight_layout が計算を狂わせることがあるので、
# レイアウト崩れが起きる場合はコメントアウトしてください
plt.show()

#==========================================================================
#==========================================================================
#==========================================================================

def visualize_AE_step(model, dataloader, device):
    print("モデル中間層の可視化")
    model.eval()
    sample_batch = next(iter(dataloader))
    sample_input = sample_batch[0][0:1]
    sample_input = sample_input.to(device)

    print(f"入力データの形状: {sample_input.shape}")

    with torch.no_grad():
        conv1 = model.encoder_conv[0:2](sample_input)
        print(f"Encoder Conv1後: {conv1.shape}")

        conv2 = model.encoder_conv[2:4](conv1)
        print(f"Encoder Conv2後: {conv2.shape}")

        latent_vec = model.encoder_Linear(conv2)
        print(f"潜在変数: {latent_vec.shape}")

        dec_Linear = model.decoder_Linear(latent_vec)
        print(f"Decoder Linear後: {dec_Linear.shape}")

        reshape = dec_Linear.view(-1, model.ch_out, model.H_out, model.W_out)
        print(f"Reshape後: {reshape.shape}")

        deconv1 = model.decoder_conv[0:2](reshape)
        print(f"Decoder Deconv1後: {deconv1.shape}")

        output = model.decoder_conv[2:](deconv1)
        print(f"最終出力: {output.shape}")

    def to_np(tensor):
        return tensor.squeeze(0).cpu().numpy()
    
    fig, axes = plt.subplots(7,1, figsize=(10,18), constrained_layout=True)
    fig.suptitle("CNN AE 中間層の変化")

    in_img = to_np(sample_input)
    axes[0].imshow(np.concatenate([in_img[0], in_img[1]], axis=0), aspect='auto', cmap='viridis')
    axes[0].set_title(f"入力: {sample_input.shape[1:]} (上: L, 下: R)")

    c1_img = to_np(conv1)
    axes[1].imshow(np.mean(c1_img, axis=0), aspect='auto', cmap='inferno')
    axes[1].set_title(f"Encoder Conv1後: {conv1.shape[1:]} (Ave)")

    c2_img = to_np(conv2)
    axes[2].imshow(np.mean(c2_img, axis=0), aspect='auto', cmap='inferno')
    axes[2].set_title(f"Encoder Conv2後: {conv2.shape[1:]} (Ave)")

    latent_data = to_np(latent_vec)
    axes[3].bar(range(len(latent_data)), latent_data, color='purple')
    axes[3].set_title(f"潜在変数: {latent_vec.shape}")
    axes[3].set_xlabel("Index")

    reshape_img = to_np(reshape)
    axes[4].imshow(np.mean(reshape_img, axis=0), aspect='auto', cmap='inferno')
    axes[4].set_title(f"Decoder Reshape後: {reshape.shape[1:]} (Ave)")

    dc1_img = to_np(deconv1)
    axes[5].imshow(np.mean(dc1_img, axis=0), aspect='auto', cmap='inferno')
    axes[5].set_title(f"Decoder Deconv1後: {dc1_img.shape[1:]} (Ave)")

    out_img = to_np(output)
    axes[6].imshow(np.concatenate([out_img[0], out_img[1]], axis=0), aspect='auto', cmap='viridis')
    axes[6].set_title(f"最終出力: {output.shape[1:]} (上: L, 下: R)")

    for ax in axes:
        if ax != axes[3]:
            ax.axis('off')

    plt.show()

if 'model' in globals() and 'dataloader_eval' in globals():
    visualize_AE_step(model, dataloader_eval, device)
else:
    print("モデルまたはデータローダーの準備ができていません")