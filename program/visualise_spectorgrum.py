import pysofaconventions
from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- 設定部分 ---
# フォント設定
matplotlib.rcParams['font.family'] = 'Yu Gothic'
# 全体のフォントサイズを大きく設定 (ここを調整するとさらに変わります)
matplotlib.rcParams['font.size'] = 14 

# 分析したい角度を指定 (度数法)
target_az = 90   # 方位角 (Azimuth): 正面0度, 左+90度, 右270度(-90度) ※SOFAの定義による
target_el = 0   # 仰角 (Elevation): 水平0度, 上+90度, 下-90度

# --- データ読み込み ---
# ファイルパスは環境に合わせて変更してください
sofa = pysofaconventions.SOFAFile('ari/hrtf_nh2.sofa', 'r')
IR = sofa.getDataIR().data
SR = sofa.getVariableValue("Data.SamplingRate")
SP = sofa.getVariableValue("SourcePosition")

# --- 角度からインデックスを検索 ---
# 指定した角度に最も近いデータのインデックスを探す (ユークリッド距離の最小値)
# SP[:, 0] が Azimuth, SP[:, 1] が Elevation
diff = np.sqrt((SP[:, 0] - target_az)**2 + (SP[:, 1] - target_el)**2)
target_idx = np.argmin(diff)

# 実際に選択された角度を取得（確認用）
selected_az = SP[target_idx, 0]
selected_el = SP[target_idx, 1]

print(f"Target: Az={target_az}, El={target_el}")
print(f"Selected Index: {target_idx} (Az={selected_az}, El={selected_el})")
print(f"Sampling Rate: {SR}")

irL = IR[target_idx, 0, :]
irR = IR[target_idx, 1, :]

# --- STFT処理 ---
f, t, Z_L = stft(irL, SR, nperseg=128, noverlap=64, window='hann', scaling='spectrum')
_, _, Z_R = stft(irR, SR, nperseg=128, noverlap=64, window='hann', scaling='spectrum')

# 転置して (時間, 周波数) の形にする
spec_L_db = 20 * np.log10(np.abs(Z_L) + 1e-12)
spec_R_db = 20 * np.log10(np.abs(Z_R) + 1e-12)

# --- プロット (ウィンドウを分ける) ---

# 左耳 (Left Ear) のウィンドウ
plt.figure(figsize=(8, 4)) # ウィンドウサイズを指定
# plt.title(f'Left Ear STFT (Az: {selected_az}°, El: {selected_el}°)', fontsize=18)
# pcolormesh(X軸=周波数, Y軸=時間)
c1 = plt.pcolormesh(t*1000, f, spec_L_db, shading='nearest', cmap='inferno')
plt.xlabel('Time [ms]', fontsize=16)
plt.ylabel('Frequency [Hz]', fontsize=16)
plt.colorbar(c1, label='Magnitude [dB]')
plt.tight_layout()

# 右耳 (Right Ear) のウィンドウ
plt.figure(figsize=(8, 4)) # ウィンドウサイズを指定
# plt.title(f'Right Ear STFT (Az: {selected_az}°, El: {selected_el}°)', fontsize=18)
c2 = plt.pcolormesh(t*1000, f, spec_R_db, shading='nearest', cmap='inferno')
plt.xlabel('Time [ms]', fontsize=16)
plt.ylabel('Frequency [Hz]', fontsize=16)
plt.colorbar(c2, label='Magnitude [dB]')
plt.tight_layout()

# 表示
plt.show()