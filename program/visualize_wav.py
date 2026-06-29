import soundfile as sf
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定
# ==========================================
wav_file_path = "WAV/piano_A4.wav" # 読み込みたいファイルパス

# STFTの設定
N_FFT = 2048       # 窓の長さ（librosaのn_fftと同じ役割）
HOP_LENGTH = 512   # ずらす幅

# ==========================================
# 2. WAV読み込み (soundfile使用)
# ==========================================
# ファイルがない場合のためにダミーデータ（指数減衰するノイズ）を作成します
try:
    data, sr = sf.read(wav_file_path)
    print(f"Loaded: {wav_file_path}, Rate: {sr}Hz")
except FileNotFoundError:
    print("ファイルが見つからないため、ダミーのインパルス応答(IR)を生成します...")
    sr = 44100
    duration = 2.0 # 秒
    t = np.linspace(0, duration, int(sr * duration))
    # ホワイトノイズ * 指数減衰 = 擬似的なIR
    noise = np.random.normal(0, 1, len(t))
    decay = np.exp(-5 * t)
    data = noise * decay

# ステレオの場合はモノラルに変換 (左右平均)
if data.ndim > 1:
    data = np.mean(data, axis=1)

# ==========================================
# 3. 短時間フーリエ変換 (scipy.signal.stft)
# ==========================================
# scipyでは overlap (重なり) のサンプル数を指定します。
# overlap = 窓長 - hop_length
n_overlap = N_FFT - HOP_LENGTH

# f: 周波数軸の配列, t_stft: 時間軸の配列, Zxx: 複素数のSTFT結果
f, t_stft, Zxx = scipy.signal.stft(data, fs=sr, window='hann', 
                                   nperseg=N_FFT, noverlap=n_overlap)

# ==========================================
# 4. 振幅スペクトルとdB変換 (手計算)
# ==========================================
# 複素数の絶対値を取って振幅にする
magnitude = np.abs(Zxx)

# dBに変換する数式: 20 * log10(振幅)
# log(0)を防ぐために微小な値(1e-6など)を足すのが定石です
ref_value = np.max(magnitude) # 最大値を基準(0dB)にするため
S_db = 20 * np.log10((magnitude + 1e-6) / ref_value)

# ==========================================
# 5. 描画 (matplotlib.pyplot)
# ==========================================
plt.figure(figsize=(10, 6))

# pcolormesh を使ってヒートマップを描く
# shading='gouraud' を指定すると滑らかに表示されます（librosaに近い見た目）
# vminを使って、ノイズフロア以下のあまりに小さい値を足切りすると綺麗に見えます（例: -80dBまで）
plt.pcolormesh(t_stft, f, S_db, shading='gouraud', cmap='inferno', vmin=-80, vmax=0)


plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Amplitude [dB]')

# 必要に応じて対数軸にする場合
plt.yscale('log')
plt.ylim(20, sr/2) # 可聴域に制限（20Hz〜ナイキスト周波数）

plt.tight_layout()
plt.show()