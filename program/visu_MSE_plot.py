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
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import stft

#========================================================================
# 設定とデバイス準備
#========================================================================
matplotlib.rcParams['font.family'] = 'Yu Gothic'
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

HRTF_DATA_DIR = 'ari'
HRTF_FILE_LIST = glob.glob(os.path.join(HRTF_DATA_DIR, '*.sofa'))
AE_EPOCHS = 50
BATCH_SIZE = 64
LATENT_DIMS_TO_TEST = [5, 10, 15, 20, 25, 30]

if not HRTF_FILE_LIST:
    print(f"Error: {HRTF_DATA_DIR}ディレクトリにsofaファイルが見つかりません")
    exit()

#========================================================================
# データ処理関数
#========================================================================
def compute_spectrogram(IR, SR, nperseg=128, noverlap=64):
    all_specs = []
    num_dir = IR.shape[0]
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

# データのロードと正規化
all_specs_for_training = []
for sofafile in HRTF_FILE_LIST:
    try:
        sofa = pysofaconventions.SOFAFile(sofafile, 'r')
        IR = sofa.getDataIR().data
        SR = int(sofa.getVariableValue("Data.SamplingRate"))
        specs, freqs, times = compute_spectrogram(IR, SR)
        all_specs_for_training.append(specs)
        print(f"Loaded: {os.path.basename(sofafile)}")
    except Exception as e:
        print(f"Skip {sofafile}: {e}")

full_spec_data = np.concatenate(all_specs_for_training, axis=0)
spec_mean, spec_std = full_spec_data.mean(), full_spec_data.std()
full_spec_data_norm = (full_spec_data - spec_mean) / (spec_std + 1e-8)

data_tensor = torch.tensor(full_spec_data_norm, dtype=torch.float32).permute(0,3,1,2).to(device)
N, C, H, W = data_tensor.shape
print(f"Input Shape: {data_tensor.shape} (N, C, H, W)")

#========================================================================
# モデル定義
#========================================================================
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
            dummy_output = self.encoder_conv(torch.zeros(1, C, H, W))
            self.ch_out, self.H_out, self.W_out = dummy_output.shape[1:]
        
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
        x = self.decoder_Linear(x)
        x = x.view(-1, self.ch_out, self.H_out, self.W_out)
        reconstructed = self.decoder_conv(x)
        return latent, reconstructed

#========================================================================
# 実験ループ：潜在次元数ごとに学習
#========================================================================
results = []
criterion = nn.MSELoss()

for l_dim in LATENT_DIMS_TO_TEST:
    print(f"\n--- 実験開始: 潜在次元数 = {l_dim} ---")
    model = CNN_AE(l_dim, C, H, W).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(AE_EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{AE_EPOCHS}], Loss: {total_loss/len(dataloader):.6f}")
    
    final_mse = total_loss / len(dataloader)
    results.append(final_mse)
    print(f"次元数 {l_dim} の最終MSE: {final_mse:.6f}")

#========================================================================
# プロット
#========================================================================
plt.figure(figsize=(8, 5))
plt.plot(LATENT_DIMS_TO_TEST, results, marker='s', color='darkorange', linewidth=2)
plt.title('潜在次元数の変化による平均二乗誤差 (MSE) の推移')
plt.xlabel('潜在次元数 (Latent Dimension)')
plt.ylabel('平均二乗誤差 (MSE)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xticks(LATENT_DIMS_TO_TEST)
plt.tight_layout()
plt.show()

print("\n全ての実験が完了しました。")