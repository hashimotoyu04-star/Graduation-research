import numpy as np
from scipy.signal import lfilter, butter
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile as sf

# --- 基本設定 ---
sr = 44100
duration = 3.0
f0 = 440  # ピアノ A4（ラ）

t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# --- 1. 基本波形（基音＋倍音構成） ---
harmonics = [1.0, 0.5, 0.3, 0.15, 0.07, 0.04]
y = np.zeros_like(t)
for n, amp in enumerate(harmonics, start=1):
    phase_shift = np.random.uniform(0, 2*np.pi)
    y += amp * np.sin(2 * np.pi * n * f0 * t + phase_shift)

# --- 2. 減衰エンベロープ（ピアノ特有の打鍵感） ---
env = np.exp(-3 * t)  # 急速に減衰する音
y *= env

# --- 3. 打鍵アタックノイズ（高域寄りの短いノイズ） ---
attack_noise = np.random.randn(len(t)) * np.exp(-100 * t)
b, a = butter(1, 0.4)
attack_noise = lfilter(b, a, attack_noise)
y += 0.02 * attack_noise

# --- 4. ボディ共鳴（ローパスフィルタで丸める） ---
b, a = butter(2, 0.2)
y = lfilter(b, a, y)

# --- 5. 正規化 ---
y = y / np.max(np.abs(y))

# 再生
sd.play(y, sr)
sd.wait()

white = np.random.randn(sr*3)
white_fft = np.fft.rfft(white)
S = np.sqrt(np.arange(white_fft.size) + 1.)
pink_spectrum = white_fft / S
pink = np.fft.irfft(pink_spectrum)

if len(pink) > len(t):
    pink = pink[:len(t)]
else:
    pink = np.pad(pink, (0, len(t) - len(pink)))
    
if np.max(np.abs(pink)) > 0:
    pink = pink / np.max(np.abs(pink)) * 0.9

sd.play(pink, sr)
sd.wait()

filename = "piano_A4.wav"
sf.write(filename, y, sr)
print(f"保存完了{filename}")

filename_pink = "pink_noise.wav"
sf.write(filename_pink, pink, sr)
print(f"保存完了{filename_pink}")
