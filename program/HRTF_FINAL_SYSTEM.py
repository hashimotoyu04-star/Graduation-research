import csv
import datetime
import glob
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
from functools import partial
import random
import numpy as np
import pandas as pd
import pysofaconventions
import soundfile as sf
import sounddevice as sd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import fftconvolve, stft, resample
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# データの可視化（散布図）と次元削減（PCA）を行うためのライブラリ
import matplotlib
matplotlib.use("TkAgg") 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

# --- 定数・設定定義 ---

HRTF_DIR = 'ari'                                        # HRTFデータ（SOFA形式）が格納されているフォルダ名
MODEL_FILE = 'CNN_weight/CNN_AE_model_2_5_65.pth'       # 学習済みモデル（重み）の保存・読み込みパス
CACHE_DIR = 'CACHE'                                    # 特徴量抽出後のデータを保存するキャッシュフォルダ
AUDIO_FILE = 'WAV/piano_A4.wav'                         # 音像定位テストに使用するドライソース（無響音声）のパス  
LOG_FILE = 'LOG.csv'                   # 実験結果を記録するCSVファイル名
RESULT_FILE = 'RESULT.csv'


N_CLUSTERS = 7                                          # 1ラウンドあたりに提示する最大音源数
AE_EPOCHS = 50                                          # オートエンコーダの学習エポック数
BATCH_SIZE = 64                                         # ミニバッチ学習のバッチサイズ
LATENT_DIM = 20                                         # オートエンコーダによって圧縮される潜在空間の次元数
MOVE_INTERVAL = 0.5                                     # 音源が移動する際の、次の角度へ遷移するまでの時間間隔（秒）
MOVE_STEPS = 12                                         # 音源移動のステップ数（一周を何分割するか）
ANGLE_STEP = 360 / MOVE_STEPS                           # 1ステップあたりの移動角度（360度 / ステップ数）
RATE_RANGE = 10                                         # ユーザーによる評価段階（1〜10）

# 再現性確保のための乱数シード固定
np.random.seed(42)
torch.manual_seed(42)

# --- ニューラルネットワーク定義 ---
# HRTFのスペクトログラム画像を圧縮・復元する畳み込みオートエンコーダ
class CNN_AE(nn.Module):
    def __init__(self, latent_dim, C, H, W):
        super(CNN_AE, self).__init__()
        # エンコーダ部分：画像特徴を抽出し、ダウンサンプリングを行う
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU()
        )
        # 畳み込み後の次元サイズを計算（Flattenのため）
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            dummy_output = self.encoder_conv(dummy_input)
            self.ch_out = dummy_output.shape[1]
            self.H_out = dummy_output.shape[2]
            self.W_out = dummy_output.shape[3]
        self.F_dim = self.ch_out * self.H_out * self.W_out
        
        # 全結合層により低次元（latent_dim）へ圧縮
        self.encoder_Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F_dim, latent_dim),
            nn.ReLU()
        )
        # デコーダ部分：圧縮された特徴を元の画像サイズへ復元
        self.decoder_Linear = nn.Sequential(
            nn.Linear(latent_dim, self.F_dim), 
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(0,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, kernel_size=(3,3), stride=(1,2), padding=(1,1), output_padding=(0,0)),
            nn.Identity() # 出力層（活性化関数なし）
        )

    def forward(self, x):
        # エンコード処理
        x = self.encoder_conv(x)
        x = self.encoder_Linear(x)
        latent = x # 潜在ベクトル（特徴量）
        
        # デコード処理
        x = self.decoder_Linear(x)
        x = x.view(-1, self.ch_out, self.H_out, self.W_out)
        reconstructed = self.decoder_conv(x)
        
        return latent, reconstructed

# --- データ処理・音声再生管理クラス ---
class HRTF_Processor:
    def __init__(self, root=None): 
        self.root = root
        # GPUが使える場合はGPUを使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hrtf_dir = HRTF_DIR
        self.model_path = MODEL_FILE
        self.batch_size = BATCH_SIZE
        self.latent_dim = LATENT_DIM
        self.ae_epochs = AE_EPOCHS

        self.is_ready = False               # データ準備完了フラグ
        self.hrtf_vec = None                # 抽出されたHRTF特徴量ベクトル

        self.status_msg = "初期化待機中"
        self.error_msg = ""

        self.valid_idx = []                 # 有効なファイルインデックス
        self.valid_file = []                # 有効なファイルパスリスト

    # バックグラウンドスレッドで実行されるデータ準備処理
    def ready_data_bg(self): 
        try:
            if not self.check_integrity():  # SOFAファイルの存在確認と整合性チェック
                return
            if not self.get_feat_vector():  # 特徴量の抽出（キャッシュ読み込み または AE学習・推論）
                return
            self.is_ready = True
        except Exception as e:
            self.error_msg = str(e)
            print(f"Data Ready Error: {e}")
    
    # SOFAファイルの読み込みテストと角度情報の整合性確認
    def check_integrity(self):
        files = glob.glob(os.path.join(self.hrtf_dir, '*.sofa'))
        if not files:
            files = glob.glob('*.sofa')
        if not files:
            self.error_msg = "No .sofa files found."
            return False
        
        base_coords = None
        valid_f = []
        # 各ファイルを開いて角度グリッドが一致しているか確認
        for i, f in enumerate(files):
            self.status_msg = f"整合性チェック中 ({i+1}/{len(files)})"
            try:
                sofa = pysofaconventions.SOFAFile(f, 'r')
                SP = sofa.getVariableValue("SourcePosition")
                SP[:,0] = np.mod(SP[:,0], 360) # アジマス角を0-360に正規化
                SP = np.round(SP, 3)

                current_coords = SP[:,0:2]

                if base_coords is None:
                    base_coords = current_coords
                    valid_f.append(f)
                    print(f"基準ファイル設定: {os.path.basename(f)}(データ点数:{len(base_coords)})")
                else:
                    # 最初のファイルと座標系が一致するものだけを採用
                    if np.array_equal(base_coords, current_coords):
                        valid_f.append(f)
                    else:
                        print(f"除外{os.path.basename(f)}")
                sofa.close()
            except:
                pass

        if len(valid_f) == 0:
            self.error_msg = "No valid SOFA files."
            return False
        
        self.valid_file = valid_f
        print(f"一致したファイル数:{len(valid_f)}")
        return True
    
    # インパルス応答(IR)からスペクトログラムを計算する静的メソッド
    @staticmethod
    def compute_spectrogrum(IR, SR):
        all_dir_spec = []
        # 各方向ごとのIRに対してSTFTを実行
        for i in range(IR.shape[0]):
            irL = IR[i,0,:]
            irR = IR[i,1,:]
            _, _, Z_L = stft(irL, SR, nperseg=128, noverlap=64, window='hann', scaling='spectrum')
            _, _, Z_R = stft(irR, SR, nperseg=128, noverlap=64, window='hann', scaling='spectrum')
            # 対数振幅スペクトル（dB）に変換
            spec_L_db = 20 * np.log10(np.abs(Z_L).T + 1e-12)
            spec_R_db = 20 * np.log10(np.abs(Z_R).T + 1e-12)
            # 左右チャンネルを結合
            one_dir_spec = np.stack([spec_L_db, spec_R_db], axis=-1)
            all_dir_spec.append(one_dir_spec)
        return np.array(all_dir_spec)

    # HRTFデータから特徴量ベクトルを取得するメイン処理
    def get_feat_vector(self): 
        # キャッシュディレクトリの作成
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR) 

        cache_file = os.path.join(CACHE_DIR, 'hrtf_vector_cache.npz')
        # キャッシュが存在し、ファイルリストが一致すれば読み込む
        if os.path.exists(cache_file):
            self.status_msg = "キャッシュを読み込み中"
            try:
                cache_d = np.load(cache_file, allow_pickle=True)
                if cache_d['file_paths'].tolist() == self.valid_file:
                    self.hrtf_vec = cache_d['features']
                    self.valid_idx = list(range(len(self.hrtf_vec)))
                    return True
            except:
                pass

        # キャッシュがない場合、SOFAからIRを読み込みスペクトログラムを作成
        all_file_spec = []
        file_counts = []
        for i, f in enumerate(self.valid_file):
            self.status_msg = f"特徴抽出中 ({i+1}/{len(self.valid_file)})"
            try:
                sofa = pysofaconventions.SOFAFile(f, 'r')
                IR = sofa.getDataIR().data
                SR = int(sofa.getVariableValue("Data.SamplingRate"))
                one_file_spec = self.compute_spectrogrum(IR, SR)
                all_file_spec.append(one_file_spec)
                file_counts.append(IR.shape[0])
                sofa.close()
            except:
                pass

        if not all_file_spec:
            self.error_msg = "Feature extraction failed."
            return False
        
        # 全データを結合し正規化
        full_data = np.concatenate(all_file_spec, axis=0)
        norm_data = (full_data - full_data.mean()) / (full_data.std() + 1e-8)
        tensor_data = torch.tensor(norm_data, dtype=torch.float32).permute(0,3,1,2).to(self.device)
        N, C, H, W = tensor_data.shape

        model = CNN_AE(self.latent_dim, C, H, W).to(self.device)

        # モデルが存在しない場合は新規学習を行う
        if not os.path.exists(self.model_path):
            dir_name = os.path.dirname(self.model_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            crit = nn.MSELoss()
            opt = optim.Adam(model.parameters(), lr=1e-3)
            dataset = TensorDataset(tensor_data, tensor_data)
            dataloader = DataLoader(dataset, self.batch_size, shuffle=True)

            model.train()
            for epoch in range(self.ae_epochs):
                total_loss = 0
                for inputs, targets in dataloader:
                    opt.zero_grad()
                    _, outputs = model(inputs)
                    loss = crit(outputs, targets)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                if (epoch+1) % 10 == 0 or epoch == 0:
                    self.status_msg = f"学習中 ({epoch+1}/{self.ae_epochs}), Loss: {total_loss/len(dataloader):.6f}"
            torch.save(model.state_dict(), self.model_path)
        else:
            # 学習済みモデルがある場合は読み込み
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except:
                self.error_msg = "Model load failed."
                return False
            
        # 学習済みモデルを使って全データの特徴量（潜在ベクトル）を抽出
        model.eval()
        latent_list = []
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                latent, _ = model(batch[0].to(self.device))
                latent_list.append(latent.cpu().numpy())

        try:
            # 抽出したベクトルをファイル単位にまとめる（ファイルごとに1つの特徴ベクトルにする）
            all_latents = np.concatenate(latent_list, axis=0)
            vec_per_file = []
            start = 0
            for count in file_counts:
                end = start + count
                vec_per_file.append(all_latents[start:end].flatten())
                start = end
            
            raw_feat = np.array(vec_per_file)
            # ベクトルのスケーリング（標準化）
            scaler = StandardScaler()
            self.hrtf_vec = scaler.fit_transform(raw_feat)
            self.valid_idx = list(range(len(self.hrtf_vec)))
            # 結果をキャッシュに保存
            np.savez(cache_file, features=self.hrtf_vec, file_paths=self.valid_file)
        except Exception as e:
            self.error_msg = f"Vectorization failed: {e}"
            return False
        return True
    
    # 再生用のドライソース（無響音）を読み込み、リサンプリングとフェード処理を行う
    def get_dry_source(self, fs_target):
        if os.path.exists(AUDIO_FILE):
            try:
                data, fs = sf.read(AUDIO_FILE)
                if fs != fs_target:
                    num_samples = int(len(data) * fs_target / fs)
                    data = resample(data, num_samples)
                if data.ndim == 2:
                    data = data[:, 0]
                fade_samples = int(0.02 * fs_target)
                if len(data) > 2 * fade_samples:
                    fade_in = np.linspace(0,1, fade_samples)
                    fade_out = np.linspace(1,0, fade_samples)
                    data[:fade_samples] *= fade_in
                    data[-fade_samples:] *= fade_out
                return data
            except Exception:
                pass
        return np.random.randn(int(fs_target * 5.0)) * 0.1

    # SOFAファイルから指定アジマス角に最も近いHRTFインパルス応答を取得
    def get_ir_from_sofa(self, sofa, azimuth):
        SP = sofa.getVariableValue("SourcePosition")
        target_az = azimuth % 360
        target_el = 0

        az_diff = np.abs(SP[:, 0] - target_az)
        az_diff = np.minimum(az_diff, 360 - az_diff)
        
        el_diff = np.abs(SP[:,1] - target_el)
        
        total_diff = az_diff + el_diff
        idx = np.argmin(total_diff)
        IR = sofa.getDataIR().data
        print(SP[idx,:])
        return IR[idx, 0, :], IR[idx, 1, :]
    
    # 音像移動シミュレーション音声を生成・再生する（スレッド実行用）
    def play_sound_thread(self, hrtf_idx):
        if self.root:
            self.root.after(0, lambda: self.root.config(cursor="watch"))
        try:
            sd.stop()
            file_path = self.valid_file[hrtf_idx]
            sofa = pysofaconventions.SOFAFile(file_path, 'r')
            SR_val = sofa.getVariableValue("Data.SamplingRate")
            SR = SR_val[0] if np.iterable(SR_val) else SR_val
            dry_data = self.get_dry_source(SR)
            
            steps = MOVE_STEPS
            interval_samples = int(MOVE_INTERVAL * SR)
            total_length = (interval_samples * steps) + len(dry_data) + 48000
            output_l = np.zeros(total_length)
            output_r = np.zeros(total_length)
            current_azimuth = 0
            
            # 各ステップごとに角度を変えながら畳み込み合成を行う
            for i in range(steps):
                hrtf_l, hrtf_r = self.get_ir_from_sofa(sofa, current_azimuth)
                conv_l = fftconvolve(dry_data, hrtf_l, mode='full')
                conv_r = fftconvolve(dry_data, hrtf_r, mode='full')
                start_pos = i * interval_samples
                end_pos = start_pos + len(conv_l)
                if end_pos > total_length:
                    valid = total_length - start_pos
                    output_l[start_pos:] += conv_l[:valid]
                    output_r[start_pos:] += conv_r[:valid]
                else:
                    output_l[start_pos : end_pos] += conv_l
                    output_r[start_pos : end_pos] += conv_r
                current_azimuth += ANGLE_STEP

            sofa.close()
            # 音割れ防止のためのノーマライズ
            max_val = max(np.max(np.abs(output_l)), np.max(np.abs(output_r)))
            if max_val > 0:
                output_l = (output_l / max_val) * 0.8
                output_r = (output_r / max_val) * 0.8
            audio_out = np.vstack([output_l, output_r]).T
            sd.play(audio_out, SR)
        except Exception as e:
            traceback.print_exc()
            if self.root:
                self.root.after(0, lambda: messagebox.showerror("Audio Error", f"再生失敗: {e}"))
        finally:
            if self.root:
                self.root.after(0, lambda: self.root.config(cursor=""))

# --- アプリケーションGUIクラス ---

class HRTF_MAIN_APP:
    def __init__(self, root):
        self.root = root
        self.root.title("HRTF 個人化")  
        self.root.state("zoomed")                          # ウィンドウのサイズを調整
        self.root.protocol("WM_DELETE_WINDOW", self.quit_APP)   # ウィンドウの×を押すとウィンドウを終了

        self.processor = HRTF_Processor(root=self.root)         # 別クラスを呼び出し      

        self.userID = tk.StringVar()
        self.current_round = 0                                  # 現在のラウンド数
        self.round_tasks = []                                   # そのラウンドにおける提案手法とランダム手法の順番リスト
        self.current_task_idx = 0                               # ラウンドタスクのリストの何番目を実行中かを示す

        self.p_candidates = []                                  # 提案手法用の候補インデックスリスト
        self.r_pool = []                                        # ランダム手法用の候補プール
        
        self.current_n_presentation = N_CLUSTERS                # 現在のラウンドでの提示音源数

        self.last_p_labels = []                                 # 今回の候補がそれぞれどのグループに分類されたかの一覧
        self.last_p_idx = []                                    # クラスタリング計算に使ったIDのリスト
        self.last_presentation_cluster_ids = []                 # 音源xがそれぞれどのグループの代表者化を記録したIDリスト

        self.current_presentation = []                          # 画面に表示中のHRTFのインデックスリスト
        
        self.selected_ui_index = tk.IntVar(value=-1)            # ユーザーがラジオボタンで選択した位置
        self.selected_score = tk.DoubleVar(value=5.0)                # ユーザーがスライダーで設定した評価点

        self.logs = []                                          # 実験ログ保存用

        # 可視化関連の変数
        self.vis_window = None
        self.vis_canvas = None
        self.ax = None
        self.fig = None
        self.pca_model = None
        self.coords_2d = None

        # データ準備をバックグラウンドで開始
        self.bg_thread = threading.Thread(target=self.processor.ready_data_bg, daemon=True)
        self.bg_thread.start()

        self.setup_login_UI()

    # ログイン画面の構築
    def setup_login_UI(self):
        self.clear_UI()
        self.login_frame = tk.Frame(self.root)
        self.login_frame.pack(expand=True)

        tk.Label(self.login_frame, text="HRTF SYSTEM", font=("Meiryo", 40, "bold")).pack(pady=13)
        desc = (
            "【実験概要】\n"
            "・提示される複数の立体音響を聞き比べ，\n"
            "・「最も聞こえが良い」ものを1つ選択してください。\n"
            "・その後、選択した音源の自然さを評価してください。\n\n"
            "【注意事項】\n"
            "・ユーザー名には実名を使用しないでください\n"
            "・一度終了すると結果は保存され，取り消しはできません\n"
            "・途中で体調が悪くなった場合は「✖」ボタンで終了してください\n"
            " --- 中断した場合，データは棄却されます ---\n"
        )
        tk.Label(self.login_frame, text=desc, font=("Meiryo", 13), justify="left", bg="#f0f0f0", padx=10, pady=10, relief="solid", bd=1).pack(pady=10)
        tk.Label(self.login_frame, text="ユーザー名", font=("Meiryo", 16)).pack()
        self.entryID = ttk.Entry(self.login_frame, textvariable=self.userID, font=("Arial", 20))
        self.entryID.pack(pady=10)
        self.entryID.focus_set()
        self.entryID.bind("<Return>", partial(self.check_start))

        self.login_btn = tk.Button(self.login_frame, text="Start", command=self.check_start, state='disabled', font=("Meiryo", 15, "bold"), width=15, height=2, relief="raised", bd=6)
        self.login_btn.pack(pady=20)

        self.label_status = tk.Label(self.login_frame, text="Loading...", fg="blue", font=("Meiryo", 14))
        self.label_status.pack()

        # データ読み込み状況を監視
        self.root.after(500, self.monitor_loading)

    # バックグラウンド処理の完了を監視し、準備が完了したらボタンを有効化
    def monitor_loading(self):
        if not self.login_frame.winfo_exists():
            return
        if self.processor.error_msg:
            self.label_status.config(text=f"Error: {self.processor.error_msg}", fg="red")
            return
        if self.processor.is_ready:
            self.label_status.config(text="Ready", fg="green")
            self.login_btn.config(state="normal")
        else:
            self.label_status.config(text=f"{self.processor.status_msg}", fg="blue")
            self.root.after(500, self.monitor_loading)

    # スタートボタン押下時の処理（ユーザー名検証と初期化）
    def check_start(self, event=None):
        name = self.userID.get().strip()
        if not name:
            messagebox.showwarning("Warning", "ユーザー名を入力してください")
            return
        if os.path.exists(RESULT_FILE):
            try:

                df = pd.read_csv(RESULT_FILE)
                if name in df['User'].astype(str).values:
                    messagebox.showwarning("Warning", f"ユーザー名 {name} は既に存在します")
                    return
            except Exception as e:
                print(f"{e}")

        # 全データに対してPCAを実行し、2次元座標を算出（可視化用）
        if self.processor.hrtf_vec is not None and len(self.processor.hrtf_vec) > 0:
            try:
                print("Calculating PCA for visualization...")
                self.pca_model = PCA(n_components=2)
                self.coords_2d = self.pca_model.fit_transform(self.processor.hrtf_vec)
                print(f"PCA Done. Shape: {self.coords_2d.shape}")
            except Exception as e:
                print(f"PCA Error: {e}")
                self.coords_2d = None

        all_idx = list(self.processor.valid_idx)            # はじめは全HRTFが候補
        self.p_candidates = list(all_idx)                   
        self.r_pool = list(all_idx)
        self.current_round = 1
        
        # 可視化ウィンドウを表示
        # self.open_visualization_window()
        self.root.update()
        
        self.start_new_round()

    # --- 可視化ウィンドウ管理 ---
    def open_visualization_window(self):
        # ウィンドウが既にある場合は最前面に表示
        if self.vis_window is not None and tk.Toplevel.winfo_exists(self.vis_window):
            self.vis_window.deiconify()
            self.vis_window.lift()
            return 
        
        # 新規ウィンドウ作成
        self.vis_window = tk.Toplevel(self.root)
        self.vis_window.title("HRTF Candidates Visualization")
        self.vis_window.geometry("600x600")
        
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.vis_canvas = FigureCanvasTkAgg(self.fig, master=self.vis_window)
        self.vis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.vis_window.lift()

    # PCA座標上に現在の候補群と提示音源をプロットして更新
    def update_visualization(self):
        if self.vis_window is None or not tk.Toplevel.winfo_exists(self.vis_window):
            return
        if self.coords_2d is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No Visualization Data\n(PCA Failed)", ha='center')
            self.vis_canvas.draw()
            return

        self.ax.clear()
        
        # 背景として全データを薄く描画
        self.ax.scatter(self.coords_2d[:,0], self.coords_2d[:,1], c='lightgray', s=15, alpha=0.5, label='Others')

        current_method = self.round_tasks[self.current_task_idx]
        
        # 現在の手法の候補プールを青色で描画
        candidates_idx = self.p_candidates if current_method == "Proposed" else self.r_pool
        if candidates_idx:
            c_idx = np.array(candidates_idx)
            self.ax.scatter(self.coords_2d[c_idx, 0], self.coords_2d[c_idx, 1], 
                            c='blue', s=30, alpha=0.6, label='Candidates Pool')

        # 今回提示されている音源を赤色で強調表示
        if self.current_presentation:
            p_idx = np.array(self.current_presentation)
            self.ax.scatter(self.coords_2d[p_idx, 0], self.coords_2d[p_idx, 1], 
                            c='red', s=120, marker='*', label='Presented Now')

        self.ax.set_title(f"Round: {self.current_round} | Method: {current_method}")
        self.ax.legend(loc='upper right', fontsize='small')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.vis_canvas.draw()
        
        self.vis_window.deiconify()
        self.vis_window.lift()
    # ---------------------------

    # 新しいラウンドの開始処理
    def start_new_round(self):
        self.clear_UI()
        
        # 候補が1つ以下になったら実験終了
        if len(self.p_candidates) <= 1:
            self.finish_experiment()
            return

        # 残り候補数に応じて今回の提示数を決定
        self.current_n_presentation = min(N_CLUSTERS, len(self.p_candidates))       # 分けたいクラスタ数と残りの候補数の少ない方を提示する数に設定
        
        print(f"--- Round {self.current_round} Start ---")
        print(f"Candidates Left: {len(self.p_candidates)}")
        print(f"N for this round: {self.current_n_presentation}")

        # 提案手法とランダム手法の順序をランダムに決定
        tasks = ["Proposed", "Random"] 
        random.shuffle(tasks)
        self.round_tasks = tasks
        self.current_task_idx = 0
        self.run_current_task()

    # 現在のタスク（提案法 or ランダム法）を実行
    def run_current_task(self):
        method = self.round_tasks[self.current_task_idx]
        idx = self.select_candidate(method)
        if not idx:
            self.finish_experiment()
            return
        
        self.current_presentation = idx
        
        # 可視化画面を更新
        # self.update_visualization()
            
        self.setup_rating_UI(method)

    # 手法に応じた音源選択ロジック
    def select_candidate(self, method):
        n = self.current_n_presentation

        if method == "Random":
            # ランダムプールからランダムにn個抽出
            if len(self.r_pool) < n:
                return list(self.r_pool)
            return random.sample(self.r_pool, n)
        
        elif method == "Proposed":
            # 提案法：K-Medoidsクラスタリングを用いて代表点を選択
            pool = self.p_candidates
            feats = self.processor.hrtf_vec[pool]

            try:
                # 候補空間をn個のクラスタに分割し、中心点（Medoid）を選ぶ
                kmed = KMedoids(n_clusters=n, metric='euclidean', method='pam', random_state=42 + self.current_round)
                labels = kmed.fit_predict(feats)
                medoid_idx_local = kmed.medoid_indices_ 

                selected = [pool[i] for i in medoid_idx_local]

                self.last_p_labels = labels  
                self.last_p_idx = np.array(pool) 
                # 選択された音源がどのクラスタIDに対応するか保存
                self.last_presentation_cluster_ids = [labels[i] for i in medoid_idx_local]

                return selected
            except Exception as e:
                print(f"KMedoids Error: {e}")
                self.last_presentation_cluster_ids = []
                return random.sample(pool, min(len(pool), n))

    # 評価画面（音源リストと再生ボタン、評価スライダー）の構築
    def setup_rating_UI(self, method):
        self.clear_UI()
        task_seq = self.current_task_idx + 1
        
        self.selected_ui_index.set(-1)      # ラジオボタンの選択状態をリセット(-1は未選択)
        self.selected_score.set(5)          # 評価スライダーを5にリセット
        
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill="x", pady=5) # 横幅いっぱいに広げる(fill="x")
        
        # デバック用(右上に現在の手法を赤文字で表示)
        # tk.Label(header_frame, text=f"[Debug] Current Method: {method}", fg="red", font=("Arial", 10)).pack(side="top", anchor="e", padx=10)

        # メインタイトル(現在のラウンド・セットを表示)
        tk.Label(header_frame, text=f"ラウンド {self.current_round}   (セット {task_seq} / 2)", 
                 font=("Arial", 16, "bold")).pack()
        tk.Label(header_frame, text="最も聞こえが良いものを1つ選び、その点数を評価してください", 
                 font=("Meiryo", 12), fg="#333").pack()
        
        main_container = tk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=5)
        
        # スクロール可能な領域の作成
        canvas = tk.Canvas(main_container)                                                      # Canvasを作成(スクロール機能を持つのはFrameではなくCanvas)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)      # スクロールバーを作成し，Canvasの縦移動(yview)と連動
        scroll_frame = tk.Frame(canvas)                                                         # ボタンなどを実際に配置する中身のフレーム
        
        # 中身のフレームのサイズが変わったら，スクロールできる範囲の更新設定
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Canvasの中にFrameをウィンドウとして埋め込む(左上(0，0)を基準)
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        # ウィンドウの横幅が変わったときに中身のフレーム幅も合わせる設定        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        # マウスホイールでスクロールできるようにする設定
        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")     # ホイールの回転量に応じて縦スクロール

        canvas.bind_all("<MouseWheel>", _on_mousewheel)     # アプリ全体でホイールの有効化

        canvas.configure(yscrollcommand=scrollbar.set)      # スクロールバーの動きをCanvasに伝える設定
        
        # 配置(Canvasを左，スクロールバーを右に置く)
        canvas.pack(side="left", fill="both", expand=True)   
        scrollbar.pack(side="right", fill="y")              
        
        # グリッド配列の重みを設定(ボタンのきれいな配置のため)
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)
        
        # 音源ごとのパネルを生成
        for i, idx in enumerate(self.current_presentation):
            row = i // 2    # 列番号(0,0,1,1,2,2,3,...)
            col = i % 2     # 行番号(0,1,0,1,0,1,0,...)
            
            # 音源1つ分の枠を作成
            item_frame = tk.Frame(scroll_frame, bd=1, relief="groove", padx=5, pady=5)
            item_frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
            item_frame.columnconfigure(0, weight=1)

            # 音源1，音源2というラベル
            tk.Label(item_frame, text=f"音源 {i+1}", font=("Arial", 9, "bold")).pack(pady=(0,2))
            
            # ボタンを押すと別スレッドでplay_sound_threadが実行
            btn = tk.Button(item_frame, text="▶ 再生", bg="#ddffff", font=("Meiryo", 9), command=partial(self.play_sound_thread, idx), width=10)
            btn.pack(pady=2)
            
            # ラジオボタン　これが押されるとself.selected_ui_indexにiが入る
            rb = tk.Radiobutton(item_frame, text="これを選択", variable=self.selected_ui_index, value=i, font=("Meiryo", 10, "bold"), fg="blue")
            rb.pack(pady=2)
        
        # 株の固定エリアの作成
        footer = tk.Frame(self.root, bd=2, relief="raised", padx=10, pady=40, bg="#f9f9f9")
        footer.pack(fill="x", side="bottom")

        # --- ★変更ここから ---

        # 1. タイトルテキストを「上」に配置
        lbl_title = tk.Label(
            footer, 
            text="選択した音源の評価:", 
            font=("Meiryo", 15), # 文字を少し大きくしました
            bg="#f9f9f9"
        )
        # pady=(0, 10) は「上は0、下は10」の余白という意味です
        lbl_title.pack(side="top", pady=(0, 15))

        # 2. スライダーと「悪い/良い」の文字を入れるためのフレームを作成
        slider_frame = tk.Frame(footer, bg="#f9f9f9")
        slider_frame.pack(side="top", pady=10)

        # 3. フレームの中に横並びで配置
        tk.Label(slider_frame, text="不自然(1)", font=("Arial", 12), bg="#f9f9f9").pack(side="left", padx=10)
        
        scale = tk.Scale(
            slider_frame, 
            variable=self.selected_score, 
            from_=1, 
            to=RATE_RANGE, 
            orient=tk.HORIZONTAL, 
            tickinterval=1, 
            length=500, 
            width=25,
            sliderlength=50,
            showvalue=True, 
            bg="#f9f9f9", 
            troughcolor="#e6e6e6",
            resolution=0.1, 
            digits=3
        )
        scale.pack(side="left", padx=10)
        
        def jump_to_click(event):
            try:
                val = scale.tk.call(scale._w, 'get', event.x, event.y)
                scale.set(val)
            except:
                pass

        scale.bind("<Button-1>", jump_to_click)

        tk.Label(slider_frame, text="自然(10)", font=("Arial", 12), bg="#f9f9f9").pack(side="left", padx=10)

        # --- ★変更ここまで ---

        # 決定ボタン
        btn_next = tk.Button(
            footer, 
            text="決定して次へ", 
            font=("Meiryo", 14, "bold"), 
            bg="#e0e0e0", 
            command=self.submit_evaluation, 
            width=20,
            relief="raised", 
            bd=5
        )
        btn_next.pack(pady=30)

    def play_sound_thread(self, idx):
        threading.Thread(target=self.processor.play_sound_thread, args=(idx,), daemon=True).start()

    # 評価結果の送信と次の処理への遷移
    def submit_evaluation(self):
        selected_ui_idx = self.selected_ui_index.get()  
        
        if selected_ui_idx == -1:                       # ラジオボタンで何も選ばずに決定を押した場合，警告を出す 
            messagebox.showwarning("未選択", "「最も聞こえが良い」音源を1つ選択してください。")
            return

        score = self.selected_score.get()
        method = self.round_tasks[self.current_task_idx]            # 現在の手法
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 現在の時刻

        # 画面上の何番目を選択したかをHRTFのインデックスに変換
        chosen_hrtf_idx = self.current_presentation[selected_ui_idx]
        
        presented_files = [os.path.basename(self.processor.valid_file[idx]) for idx in self.current_presentation]
        
        # ログデータに追加
        self.logs.append({
            "User": self.userID.get(),
            "Round": self.current_round,
            "Method": method,
            "Selected_File": os.path.basename(self.processor.valid_file[chosen_hrtf_idx]),
            "Score": score,
            "Presented_Files": str(presented_files),
            "Time": ts
        })
            
        # 提案法の場合：選ばれたクラスタに属する音源のみを次のラウンドの候補にする
        if method == "Proposed":
            if self.last_presentation_cluster_ids:      # 前回保存した画面上のボタンのクラスターラベルを使用
                target_cluster_id = self.last_presentation_cluster_ids[selected_ui_idx]

                # 候補のラベル配列を見て選ばれたクラスターラベルと同じものだけを抽出
                mask = (self.last_p_labels == target_cluster_id)
                new_candidates = self.last_p_idx[mask].tolist()
                
                # 候補リストを更新
                self.p_candidates = new_candidates
                print(f"Proposed: Cluster {target_cluster_id} selected. Candidates {len(self.last_p_idx)} -> {len(self.p_candidates)}")
            else:
                pass

        # ランダム法の場合：重複提示を避けるため、今回提示した音源をプールから除外
        elif method == "Random":
            for used_idx in self.current_presentation:
                if used_idx in self.r_pool:
                    self.r_pool.remove(used_idx)
            print(f"Random: Remaining pool size {len(self.r_pool)}")

        self.current_task_idx += 1
        
        if self.current_task_idx < len(self.round_tasks):
            self.run_current_task()
        else:
            self.current_round += 1
            self.start_new_round()

    # 実験終了時の処理（結果保存とメッセージ表示）
    def finish_experiment(self):
        self.clear_UI()
        
        try:
            # --- 1. 既存の全ログ保存処理 (変更なし) ---
            df = pd.DataFrame(self.logs)
            
            is_exist = os.path.exists(LOG_FILE)
            cols = ["User", "Round", "Method", "Selected_File", "Score", "Presented_Files", "Time"]
            if not df.empty:
                df_save = df.reindex(columns=cols)
                df_save.to_csv(LOG_FILE, mode='a', header=not is_exist, index=False)
            
            # --- 2. 集計データの作成と保存 (新規追加) ---
            
            # (A) 提案手法の最終結果の特定
            prop_best_file = "N/A"
            prop_score = 0
            
            # 最終的に候補リスト(p_candidates)に残っているものを「提案手法の解」とする
            if self.p_candidates:
                # 候補リストの先頭（あるいは唯一残ったもの）を取得
                final_idx = self.p_candidates[0]
                prop_best_file = os.path.basename(self.processor.valid_file[final_idx])
            
            # 提案手法における最後の評価スコアを取得
            df_prop = df[df["Method"] == "Proposed"]
            if not df_prop.empty:
                # 最後の行（最終ラウンドの選択時）のスコアを採用
                prop_score = df_prop.iloc[-1]["Score"]
                # もし候補リストが空でログだけある場合は、最後に選んだファイルを解とする（念のため）
                if prop_best_file == "N/A":
                     prop_best_file = df_prop.iloc[-1]["Selected_File"]

            # (B) ランダム手法のベスト結果の特定
            rand_best_file = "N/A"
            rand_best_score = 0
            
            df_rand = df[df["Method"] == "Random"]
            if not df_rand.empty:
                # 最高スコアを見つける
                rand_best_score = df_rand["Score"].max()
                # 最高スコアを獲得した行をすべて抽出
                best_rows = df_rand[df_rand["Score"] == rand_best_score]
                # ファイル名のリストを取得（重複排除）
                best_files_list = best_rows["Selected_File"].unique().tolist()
                # 複数ある場合はセミコロンで結合して文字列にする
                rand_best_file = ";".join(best_files_list)

            # (C) サマリーデータの作成
            summary_data = {
                "User": self.userID.get(),
                "Proposed_Best_File": prop_best_file,
                "Proposed_Score": prop_score,
                "Random_Best_File": rand_best_file,
                "Random_Best_Score": rand_best_score,
                "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # (D) サマリーCSVへの保存
            df_summary = pd.DataFrame([summary_data])
            is_summary_exist = os.path.exists(RESULT_FILE)
            # カラム順序の指定（見やすくするため）
            summary_cols = ["User", "Proposed_Best_File", "Proposed_Score", "Random_Best_File", "Random_Best_Score", "Time"]
            df_summary = df_summary.reindex(columns=summary_cols)
            
            df_summary.to_csv(RESULT_FILE, mode='a', header=not is_summary_exist, index=False)


            # --- 3. 終了メッセージの表示 (表示用に整形) ---
            msg = (
                "実験終了です。\nご協力ありがとうございました。"
            )

        except Exception as e:
            msg = f"保存または集計中にエラーが発生しました: {e}\n管理者を呼んでください。"
            traceback.print_exc()
            
        result_frame = tk.Frame(self.root)
        result_frame.pack(expand=True)
        tk.Label(result_frame, text=msg, font=("Meiryo", 14), justify="left", bg="white", relief="solid", padx=20, pady=20).pack(pady=30)
        ttk.Button(result_frame, text="終了", command=self.root.destroy).pack(ipadx=20, ipady=10)

    # アプリ終了時の確認とクリーンアップ
    def quit_APP(self):
        if messagebox.askokcancel("Quit", "終了しますか？"):
            sd.stop()
            try:
                if self.vis_window:
                    self.vis_window.destroy()
            except:
                pass
            self.root.destroy()
            sys.exit()
    
    # UI要素の消去
    def clear_UI(self):
        for w in self.root.winfo_children():
            if w != self.vis_window: # 可視化ウィンドウは消さない
                w.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HRTF_MAIN_APP(root)
    root.mainloop()