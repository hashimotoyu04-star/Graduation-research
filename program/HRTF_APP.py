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
                df = pd.DataFrame({'az': SP[:,0], 'el': SP[:,1]})
                df = df.sort_values(by=['az', 'el'])
                sorted_SP = df[['az', 'el']].values

                if base_coords is None:
                    base_coords = sorted_SP
                    valid_f.append(f)
                else:
                    # 最初のファイルと座標系が一致するものだけを採用
                    if np.array_equal(base_coords, sorted_SP):
                        valid_f.append(f)
                sofa.close()
            except:
                pass

        if len(valid_f) == 0:
            self.error_msg = "No valid SOFA files."
            return False
        
        self.valid_file = valid_f
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
        self.root.state("zoomed")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_APP)

        # ※ HRTF_Processorは外部で定義されている前提
        self.processor = HRTF_Processor(root=self.root)       

        self.userID = tk.StringVar()
        self.current_round = 0
        self.round_tasks = []
        self.current_task_idx = 0

        self.p_candidates = []                                  # 提案手法用の候補リスト
        self.r_pool = []                                        # ランダム手法用のプール
        
        self.current_n_presentation = N_CLUSTERS

        self.last_p_labels = []
        self.last_p_idx = []
        self.last_presentation_cluster_ids = []

        self.current_presentation = []                          # 画面に表示中の音源インデックス
        
        self.selected_ui_index = tk.IntVar(value=-1)            # ユーザー選択 (ラジオボタン)
        self.logs = []                                          # 実験ログ保存用

        # データ準備をバックグラウンドで開始
        self.bg_thread = threading.Thread(target=self.processor.ready_data_bg, daemon=True)
        self.bg_thread.start()

        self.setup_login_UI()

    # ---------------------------------------------------------
    # ログイン画面
    # ---------------------------------------------------------
    def setup_login_UI(self):
        self.clear_UI()
        self.login_frame = tk.Frame(self.root)
        self.login_frame.pack(expand=True)

        tk.Label(self.login_frame, text="HRTF SYSTEM", font=("Meiryo", 40, "bold")).pack(pady=13)
        desc = (
            "【実験概要】\n"
            "・提示される複数の立体音響を聞き比べ，\n"
            "・「最も聞こえが良い」ものを1つ選択してください。\n\n"
            "【注意事項】\n"
            "・ユーザー名には実名を使用しないでください\n"
            "・一度終了すると結果は保存され，取り消しはできません\n"
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

        self.root.after(500, self.monitor_loading)

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

    # ---------------------------------------------------------
    # 実験開始処理（可視化計算削除済み）
    # ---------------------------------------------------------
    def check_start(self, event=None):
        name = self.userID.get().strip()
        if not name:
            messagebox.showwarning("Warning", "ユーザー名を入力してください")
            return
        
        # 既存ユーザーチェック
        if os.path.exists(RESULT_FILE):
            try:
                df = pd.read_csv(RESULT_FILE)
                if name in df['User'].astype(str).values:
                    messagebox.showwarning("Warning", f"ユーザー名 {name} は既に存在します")
                    return
            except Exception as e:
                print(f"{e}")

        # データ初期化
        all_idx = list(self.processor.valid_idx)
        self.p_candidates = list(all_idx)
        self.r_pool = list(all_idx)
        self.current_round = 1
        
        self.start_new_round()

    # ---------------------------------------------------------
    # ラウンド管理
    # ---------------------------------------------------------
    def start_new_round(self):
        self.clear_UI()
        
        # 候補が1つ以下になったら実験終了
        if len(self.p_candidates) <= 1:
            self.finish_experiment()
            return

        self.current_n_presentation = min(N_CLUSTERS, len(self.p_candidates))
        
        print(f"--- Round {self.current_round} Start ---")
        print(f"Candidates Left: {len(self.p_candidates)}")

        # 手法の順序決定
        tasks = ["Proposed", "Random"] 
        random.shuffle(tasks)
        self.round_tasks = tasks
        self.current_task_idx = 0
        self.run_current_task()

    def run_current_task(self):
        method = self.round_tasks[self.current_task_idx]
        idx = self.select_candidate(method)
        
        # 候補取得に失敗、あるいは候補切れの場合
        if not idx:
            self.finish_experiment()
            return
        
        self.current_presentation = idx
        self.setup_rating_UI(method)

    def select_candidate(self, method):
        n = self.current_n_presentation

        if method == "Random":
            # ランダムプールから選択
            if len(self.r_pool) < n:
                return list(self.r_pool)
            return random.sample(self.r_pool, n)
        
        elif method == "Proposed":
            # クラスタリングによる選択（可視化はしないが計算は行う）
            pool = self.p_candidates
            feats = self.processor.hrtf_vec[pool]
            try:
                # K-Medoids Clustering
                kmed = KMedoids(n_clusters=n, metric='euclidean', method='pam', random_state=42 + self.current_round)
                labels = kmed.fit_predict(feats)
                medoid_idx_local = kmed.medoid_indices_ 

                selected = [pool[i] for i in medoid_idx_local]

                self.last_p_labels = labels  
                self.last_p_idx = np.array(pool) 
                self.last_presentation_cluster_ids = [labels[i] for i in medoid_idx_local]
                return selected
            except Exception as e:
                print(f"KMedoids Error: {e}")
                self.last_presentation_cluster_ids = []
                return random.sample(pool, min(len(pool), n))

    # ---------------------------------------------------------
    # 評価UI (可視化・スライダーなし)
    # ---------------------------------------------------------
    def setup_rating_UI(self, method):
        self.clear_UI()
        task_seq = self.current_task_idx + 1
        
        self.selected_ui_index.set(-1)
        
        # --- ヘッダー ---
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill="x", pady=5)
        
        tk.Label(header_frame, text=f"ラウンド {self.current_round}   (セット {task_seq} / 2)", 
                 font=("Arial", 16, "bold")).pack()
        tk.Label(header_frame, text="最も聞こえが良いものを1つ選び、決定してください", 
                 font=("Meiryo", 12), fg="#333").pack()
        
        # --- メインエリア（スクロール対応） ---
        main_container = tk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=5)
        
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)   
        scrollbar.pack(side="right", fill="y")              
        
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)
        
        # --- 音源パネル生成 ---
        for i, idx in enumerate(self.current_presentation):
            row = i // 2
            col = i % 2
            item_frame = tk.Frame(scroll_frame, bd=1, relief="groove", padx=5, pady=5)
            item_frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
            item_frame.columnconfigure(0, weight=1)

            tk.Label(item_frame, text=f"音源 {i+1}", font=("Arial", 9, "bold")).pack(pady=(0,2))
            
            # 再生ボタン
            btn = tk.Button(item_frame, text="▶ 再生", bg="#ddffff", font=("Meiryo", 9), 
                            command=partial(self.play_sound_thread, idx), width=10)
            btn.pack(pady=2)
            
            # 選択ラジオボタン
            rb = tk.Radiobutton(item_frame, text="これを選択", variable=self.selected_ui_index, 
                                value=i, font=("Meiryo", 10, "bold"), fg="blue")
            rb.pack(pady=2)
        
        # --- フッター (決定ボタンのみ) ---
        footer = tk.Frame(self.root, bd=0, padx=10, pady=20, bg="#f0f0f0")
        footer.pack(fill="x", side="bottom")

        btn_next = tk.Button(
            footer, text="決定して次へ", font=("Meiryo", 14, "bold"), 
            bg="#e0e0e0", command=self.submit_evaluation, width=20, relief="raised", bd=5
        )
        btn_next.pack(pady=10)

    def play_sound_thread(self, idx):
        threading.Thread(target=self.processor.play_sound_thread, args=(idx,), daemon=True).start()

    # ---------------------------------------------------------
    # 選択結果の送信
    # ---------------------------------------------------------
    def submit_evaluation(self):
        selected_ui_idx = self.selected_ui_index.get()  
        
        if selected_ui_idx == -1:
            messagebox.showwarning("未選択", "音源を1つ選択してください。")
            return

        method = self.round_tasks[self.current_task_idx]
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chosen_hrtf_idx = self.current_presentation[selected_ui_idx]
        presented_files = [os.path.basename(self.processor.valid_file[idx]) for idx in self.current_presentation]
        
        # ログ保存 (Score除外)
        self.logs.append({
            "User": self.userID.get(),
            "Round": self.current_round,
            "Method": method,
            "Selected_File": os.path.basename(self.processor.valid_file[chosen_hrtf_idx]),
            "Presented_Files": str(presented_files),
            "Time": ts
        })
            
        # 次ラウンド候補の更新
        if method == "Proposed":
            if self.last_presentation_cluster_ids:
                target_cluster_id = self.last_presentation_cluster_ids[selected_ui_idx]
                mask = (self.last_p_labels == target_cluster_id)
                new_candidates = self.last_p_idx[mask].tolist()
                self.p_candidates = new_candidates
                print(f"Proposed: Selected Cluster {target_cluster_id}. Remaining: {len(self.p_candidates)}")

        elif method == "Random":
            for used_idx in self.current_presentation:
                if used_idx in self.r_pool:
                    self.r_pool.remove(used_idx)

        self.current_task_idx += 1
        if self.current_task_idx < len(self.round_tasks):
            self.run_current_task()
        else:
            self.current_round += 1
            self.start_new_round()

    # ---------------------------------------------------------
    # 終了処理
    # ---------------------------------------------------------
    def finish_experiment(self):
        self.clear_UI()
        
        try:
            # 1. ログ保存
            df = pd.DataFrame(self.logs)
            cols = ["User", "Round", "Method", "Selected_File", "Presented_Files", "Time"]
            if not df.empty:
                is_exist = os.path.exists(LOG_FILE)
                available_cols = [c for c in cols if c in df.columns]
                df_save = df.reindex(columns=available_cols)
                df_save.to_csv(LOG_FILE, mode='a', header=not is_exist, index=False)
            
            # 2. 結果集計 (提案手法の最終結果のみ)
            prop_final_file = "N/A"
            if self.p_candidates:
                final_idx = self.p_candidates[0]
                prop_final_file = os.path.basename(self.processor.valid_file[final_idx])
            else:
                # 念のためのフォールバック
                df_prop = df[df["Method"] == "Proposed"]
                if not df_prop.empty:
                    prop_final_file = df_prop.iloc[-1]["Selected_File"]

            summary_data = {
                "User": self.userID.get(),
                "Proposed_Result_File": prop_final_file,
                "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            df_summary = pd.DataFrame([summary_data])
            summary_cols = ["User", "Proposed_Result_File", "Time"]
            df_summary = df_summary.reindex(columns=summary_cols)
            
            is_summary_exist = os.path.exists(RESULT_FILE)
            df_summary.to_csv(RESULT_FILE, mode='a', header=not is_summary_exist, index=False)

            msg = (
                "実験終了です。\n"
                f"あなたの耳に最も適したHRTFは以下の通り推定されました：\n\n{prop_final_file}\n\n"
                "ご協力ありがとうございました。"
            )

        except Exception as e:
            msg = f"保存エラー: {e}\n管理者を呼んでください。"
            import traceback
            traceback.print_exc()
            
        result_frame = tk.Frame(self.root)
        result_frame.pack(expand=True)
        tk.Label(result_frame, text=msg, font=("Meiryo", 14), justify="center", bg="white", relief="solid", padx=20, pady=20).pack(pady=30)
        ttk.Button(result_frame, text="終了", command=self.root.destroy).pack(ipadx=20, ipady=10)

    def quit_APP(self):
        if messagebox.askokcancel("Quit", "終了しますか？"):
            self.root.destroy()
            sys.exit()

    def clear_UI(self):
        for w in self.root.winfo_children():
            w.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HRTF_MAIN_APP(root)
    root.mainloop()