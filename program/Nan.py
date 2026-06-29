import pysofaconventions
import numpy as np

def check_sofa_missing_values(file_path):
    """
    SOFAファイル内の各座標におけるインパルス応答(IR)の欠損値をチェックします。
    """
    try:
        # SOFAファイルを読み込み
        sofa = pysofaconventions.SOFAFile(file_path, 'r')
        
        # 座標データ (SourcePosition) と 音響データ (Data.IR) を取得
        # 通常 SOFAでは [M, C] や [M, R, N] のような形状をしています
        # M: 測定点数, R: レシーバー数(耳), N: サンプル数
        SP = sofa.getVariableValue('SourcePosition')
        IR = sofa.getDataIR().data
        
        print(f"ファイル読み込み成功: {file_path}")
        print(f"測定点数: {IR.shape[0]}, 座標データの形状: {SP.shape}")
        print("-" * 30)

        missing_count = 0
        
        # 各測定点(Measurement)ごとにループ
        for i in range(IR.shape[0]):
            current_ir = IR[i]
            current_pos = SP[i]
            
            # 判定フラグ
            has_nan = np.isnan(current_ir).any()
            has_inf = np.isinf(current_ir).any()
            is_all_zeros = np.all(current_ir == 0)
            
            if has_nan or has_inf or is_all_zeros:
                missing_count += 1
                status = []
                if has_nan: status.append("NaN検出")
                if has_inf: status.append("Inf検出")
                if is_all_zeros: status.append("全データ0(無音)")
                
                print(f"インデックス [{i}] で異常を検出:")
                print(f"  座標 (Az, El, R等): {current_pos}")
                print(f"  状態: {', '.join(status)}")

        if missing_count == 0:
            print("欠損値や異常データは見つかりませんでした。")
        else:
            print("-" * 30)
            print(f"合計 {missing_count} 箇所の座標で異常が見つかりました。")

        sofa.close()

    except Exception as e:
        print(f"エラーが発生しました: {e}")

# 使用例
check_sofa_missing_values('ari/hrtf%20b_nh778.sofa')