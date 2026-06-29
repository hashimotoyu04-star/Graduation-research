import glob
import os
import numpy as np
import pysofaconventions
import pandas as pd

# ---------------- 設定 ----------------
HRTF_DIR = 'ari'  # HRTFファイルが入っているフォルダ名
# --------------------------------------

def normalize_coords(sp):
    """
    座標を比較用に正規化する関数
    - アジマスを0-360に統一
    - 小数点以下3桁で丸める（浮動小数点の微細な誤差を無視するため）
    """
    sp_norm = sp.copy()
    sp_norm[:, 0] = np.mod(sp_norm[:, 0], 360)
    sp_norm = np.round(sp_norm, 3)
    return sp_norm

def check_hrtf_order():
    # ファイルリスト取得
    files = glob.glob(os.path.join(HRTF_DIR, '*.sofa'))
    if not files:
        # フォルダ指定がない場合、カレントディレクトリも探す
        files = glob.glob('*.sofa')
    
    if not files:
        print(f"エラー: '{HRTF_DIR}' フォルダまたはカレントディレクトリに .sofa ファイルが見つかりません。")
        return

    print(f"対象ファイル数: {len(files)}")
    print("チェックを開始します...\n")

    # 基準となる最初のファイルを読み込む
    base_file = files[0]
    try:
        sofa_base = pysofaconventions.SOFAFile(base_file, 'r')
        base_raw_pos = sofa_base.getVariableValue("SourcePosition")
        base_pos = normalize_coords(base_raw_pos)
        sofa_base.close()
        print(f"基準ファイル: {os.path.basename(base_file)}")
        print(f"基準データ点数: {base_pos.shape[0]} 点")
    except Exception as e:
        print(f"基準ファイルの読み込みに失敗しました: {e}")
        return

    mismatch_files = []
    error_files = []

    # 2つ目以降のファイルと比較
    for f in files[1:]:
        fname = os.path.basename(f)
        try:
            sofa = pysofaconventions.SOFAFile(f, 'r')
            curr_raw_pos = sofa.getVariableValue("SourcePosition")
            curr_pos = normalize_coords(curr_raw_pos)
            sofa.close()

            # 1. データ点数のチェック
            if base_pos.shape != curr_pos.shape:
                print(f"[NG] {fname}: データ点数が異なります (基準:{base_pos.shape[0]}, このファイル:{curr_pos.shape[0]})")
                mismatch_files.append(fname)
                continue

            # 2. 並び順を含めた完全一致チェック
            # np.array_equal は要素の順序も同じである必要があります
            if not np.array_equal(base_pos, curr_pos):
                print(f"[NG] {fname}: 座標の並び順、または値が一致しません")
                
                # 詳細な理由を表示（デバッグ用）
                # 試しにソートして一致するか確認（ソートして一致するなら、単に順番違い）
                df_base = pd.DataFrame(base_pos, columns=['az', 'el', 'r'])
                df_curr = pd.DataFrame(curr_pos, columns=['az', 'el', 'r'])
                
                df_base_sorted = df_base.sort_values(by=['az', 'el']).values
                df_curr_sorted = df_curr.sort_values(by=['az', 'el']).values
                
                if np.array_equal(df_base_sorted, df_curr_sorted):
                    print("   -> 備考: ソートすれば一致します（順序だけが違います）")
                else:
                    print("   -> 備考: 含まれている座標セット自体が異なります")
                
                mismatch_files.append(fname)

        except Exception as e:
            print(f"[Error] {fname}: 読み込みエラー ({e})")
            error_files.append(fname)

    # --- 結果発表 ---
    print("\n" + "="*30)
    print("       検証結果サマリー")
    print("="*30)

    if not mismatch_files and not error_files:
        print("✅ OK: 全てのファイルの座標順序は完全に一致しています。")
        print("今のコード（HRTF_FINAL_SYSTEM.py）をそのまま使用しても問題ありません。")
    else:
        print(f"❌ NG: 問題のあるファイルが見つかりました ({len(mismatch_files) + len(error_files)} 件)")
        if mismatch_files:
            print("\n以下のファイルは基準ファイルと座標順序または内容が異なります:")
            for mf in mismatch_files:
                print(f" - {mf}")
            print("\n【対策】")
            print("1. データセットを作り直して順序を統一する")
            print("2. または、メインコード(HRTF_FINAL_SYSTEM.py)に「読み込み時のソート処理」を追加する")
        
        if error_files:
            print("\n読み込めなかったファイル:")
            for ef in error_files:
                print(f" - {ef}")

if __name__ == "__main__":
    check_hrtf_order()
    