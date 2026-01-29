import pandas as pd
import os

# ファイル名とフォルダ名の設定
input_folder = 'CSV'
output_folder = 'Result'
input_file_name = '20251201A84_Trace_log.csv'
output_file_name = '20251201A84_Trace_log_cleaned.csv'

# パスの結合
input_path = os.path.join(input_folder, input_file_name)
output_path = os.path.join(output_folder, output_file_name)

# 保存先のフォルダが存在しない場合は作成する
os.makedirs(output_folder, exist_ok=True)

# CSVファイルの読み込み
try:
    df = pd.read_csv(input_path)
    print(f"ファイルを読み込みました: {input_path}")
    print(f"処理前の行数: {len(df)}")
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。{input_path} にファイルがあるか確認してください。")
    exit()

# 重複行の削除 (keep='first' は最初の行を残して重複を削除します)
df_cleaned = df.drop_duplicates(keep='first')

# 削除された行数の計算
dropped_count = len(df) - len(df_cleaned)

# 結果の保存
df_cleaned.to_csv(output_path, index=False)

print(f"削除された重複行数: {dropped_count}")
print(f"処理後の行数: {len(df_cleaned)}")
print(f"処理後のファイルを保存しました: {output_path}")