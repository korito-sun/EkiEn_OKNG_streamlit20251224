import pandas as pd
import os

# ファイル名とフォルダ名の設定
input_folder = 'CSV'
output_folder = 'Result'
input_file_name = '20251201A84_Trace_log.csv'
output_file_name = '20251201A84_Trace_log_cleaned.csv'

# パスの結合（OSに依存しないパスを作成します）
input_path = os.path.join(input_folder, input_file_name)
output_path = os.path.join(output_folder, output_file_name)

# 保存先のフォルダが存在しない場合は作成する
os.makedirs(output_folder, exist_ok=True)

# CSVファイルの読み込み
try:
    df = pd.read_csv(input_path)
    print(f"ファイルを読み込みました: {input_path}")
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。{input_path} にファイルがあるか確認してください。")
    exit()

# 重複している列を特定して削除する関数
def remove_duplicate_columns(df):
    columns_to_drop = set()
    
    # 全列を総当たりで比較
    for i in range(df.shape[1]):
        col1_name = df.columns[i]
        
        if col1_name in columns_to_drop:
            continue
            
        col1 = df.iloc[:, i]
        
        for j in range(i + 1, df.shape[1]):
            col2_name = df.columns[j]
            
            if col2_name in columns_to_drop:
                continue
                
            col2 = df.iloc[:, j]
            
            # 値が完全に一致する場合、削除リストに追加
            if col1.equals(col2):
                columns_to_drop.add(col2_name)
    
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned, columns_to_drop

# 重複列の削除実行
df_cleaned, dropped_cols = remove_duplicate_columns(df)

# 結果の保存
df_cleaned.to_csv(output_path, index=False)

print(f"削除された列: {dropped_cols}")
print(f"処理後のファイルを保存しました: {output_path}")