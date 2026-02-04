import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def main():
    # メインウィンドウの作成（表示はしない）
    root = tk.Tk()
    root.withdraw()
    
    # 1. 読み込むCSVファイルを選択するダイアログを表示
    input_path = filedialog.askopenfilename(
        title="重複削除するCSVファイルを選択してください",
        filetypes=[("CSVファイル", "*.csv"), ("すべてのファイル", "*.*")]
    )
    
    # キャンセルされた場合は終了
    if not input_path:
        return

    try:
        # CSV読み込み
        df = pd.read_csv(input_path)
        original_rows = len(df)
        
        # 重複行の削除
        df_cleaned = df.drop_duplicates(keep='first')
        cleaned_rows = len(df_cleaned)
        dropped_rows = original_rows - cleaned_rows
        
        # 保存用ファイル名の作成（例: file.csv -> file_cleaned.csv）
        dir_name = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        file_root, file_ext = os.path.splitext(base_name)
        default_output_name = f"{file_root}_cleaned{file_ext}"
        
        # 2. 保存先を選択するダイアログを表示
        output_path = filedialog.asksaveasfilename(
            title="保存先を指定してください",
            initialdir=dir_name,
            initialfile=default_output_name,
            defaultextension=".csv",
            filetypes=[("CSVファイル", "*.csv")]
        )
        
        # キャンセルされた場合は終了
        if not output_path:
            return

        # 保存実行
        df_cleaned.to_csv(output_path, index=False)
        
        # 3. 完了メッセージを表示
        messagebox.showinfo(
            "処理完了", 
            f"完了しました。\n\n"
            f"元の行数: {original_rows}\n"
            f"処理後の行数: {cleaned_rows}\n"
            f"削除された重複行: {dropped_rows}\n\n"
            f"保存先:\n{output_path}"
        )

    except Exception as e:
        # エラーが発生した場合のメッセージ
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")

if __name__ == "__main__":
    main()