import pysofaconventions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf

# --- 1. 設定部分 ---
matplotlib.rcParams['font.family'] = 'Yu Gothic'
matplotlib.rcParams['font.size'] = 12 

# 解析・表示設定
target_az = 90  # 方位角 (左真横)
target_el = 0   # 仰角 (水平)

# --- 2. HRTF データの処理 ---
# ※ファイルパスはご自身の環境に合わせて修正してください
sofa_path = 'ari/hrtf_nh2.sofa'
sofa = pysofaconventions.SOFAFile(sofa_path, 'r')
IR_data = sofa.getDataIR().data
SR_sofa = sofa.getVariableValue("Data.SamplingRate")
SP = sofa.getVariableValue("SourcePosition")

# 最も近い角度のインデックスを検索
diff = np.sqrt((SP[:, 0] - target_az)**2 + (SP[:, 1] - target_el)**2)
target_idx = np.argmin(diff)
selected_az = SP[target_idx, 0]
selected_el = SP[target_idx, 1]

irL = IR_data[target_idx, 0, :]
irR = IR_data[target_idx, 1, :]

# HRTFのFFT解析
n_hrtf = len(irL)
freq_hrtf = np.fft.rfftfreq(n_hrtf, d=1/SR_sofa)
spec_L_db = 20 * np.log10(np.abs(np.fft.rfft(irL)) + 1e-12)
spec_R_db = 20 * np.log10(np.abs(np.fft.rfft(irR)) + 1e-12)

# --- 3. 生成済みWAVデータの処理 ---
# 前のステップで保存したファイルを読み込みます
try:
    piano_y, sr_p = sf.read("WAV/piano_A4.wav")
except FileNotFoundError:
    print("エラー: wavファイルが見つかりません。先に音声生成コードを実行してください。")
    exit()

def get_magnitude_spectrum(signal, sr):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/sr)
    magnitude = np.abs(np.fft.rfft(signal))
    # 最大値で正規化して0dB基準にする
    magnitude_db = 20 * np.log10(magnitude / (np.max(magnitude) + 1e-12) + 1e-12)
    return freq, magnitude_db

freq_piano, spec_piano = get_magnitude_spectrum(piano_y, sr_p)

# --- 4. プロットの作成 ---
plt.figure(figsize=(11, 5))

# 上段: HRTFのスペクトル (耳のフィルター特性)
plt.plot(freq_hrtf, spec_L_db, label=f'左耳 (方位角:{selected_az}°)', alpha=0.8, color='blue')
plt.plot(freq_hrtf, spec_R_db, label=f'右耳 (方位角:{selected_az}°)', alpha=0.8, color='red')
plt.xscale('log')
plt.xlabel('周波数 [Hz]')
plt.ylabel('振幅 [dB]')
plt.xlim(100,20000)
plt.plot(freq_piano, spec_piano, label="ピアノ音", alpha=0.8, color="green")
plt.legend()

plt.tight_layout()
plt.show()

print("全工程が完了しました。")