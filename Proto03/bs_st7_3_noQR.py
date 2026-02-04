# --- 修正ポイント: "nan nan" を "QR_NG" に置換 ---2025-01-08
# --- 修正: 1行ごとに色を変える（シマウマ模様） ---2025-01-08
# --- 修正: パレート図の棒グラフの上に合計数を表示 ---2025-01-09修正版
# --- 修正: st.dataframeの警告対応 (width='stretch') ---2025-01-09
# --- 修正: 期間選択を「日付範囲選択(カレンダー)」に変更し年またぎに対応 ---2025-01-22
# --- 修正: date_inputのSessionState警告エラーを解消 ---2025-02-04
# --- 修正: ファイル選択をGUI(ファイルアップローダー)に変更 ---2025-02-04追加

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import os
import matplotlib
import datetime

# 日本語フォント設定
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ページ設定
st.set_page_config(page_title="Trace Log Analysis", layout="wide")

# タイトル
st.title("液晶演出生産時検査機OK/NGダッシュボード")

# --- 修正: ハードコードされたファイルパス設定を削除 ---
# CSV_DIR = './CSV'
# CSV_FILENAME = '...' 
# FILE_PATH = ... (削除)

# データ読み込みとキャッシュ
# 注意: ファイルアップローダーから渡されるオブジェクトもpd.read_csvで読み込めます
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        
        # 重複データの削除
        initial_count = len(df)
        df = df.drop_duplicates()
        dedup_count = len(df)
        removed_count = initial_count - dedup_count
        
        # クリーニング
        target_cols = ['Model', 'FCT_ID', 'QRresult', 'TESTresult', 'TestNo.', 'ErrorNo.', 'PCB_Name', 'DateTime']
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = ''
        
        # --- 日付情報の作成 ---
        if 'DateTime' in df.columns:
            df['DateTime_dt'] = pd.to_datetime(df['DateTime'], errors='coerce')
            df['Month'] = df['DateTime_dt'].dt.strftime('%Y-%m')
            df['Date'] = df['DateTime_dt'].dt.date
            
            # --- 年と月の列 (フィルター表示用などに保持) ---
            df['Year_str'] = df['DateTime_dt'].dt.year.fillna(0).astype(int).astype(str)
            df['Month_int'] = df['DateTime_dt'].dt.month.fillna(0).astype(int)
        else:
            df['Month'] = 'Unknown'
            df['Date'] = None
            df['Year_str'] = '0'
            df['Month_int'] = 0

        # --- 判定ロジック (Status) ---
        def determine_status(row):
            qr = row.get('QRresult', '').upper()
            test = row.get('TESTresult', '').upper()
            is_qr_ok = (qr == 'OK' or qr in ['', 'NAN', 'NONE'])
            return 'OK' if (is_qr_ok and test == 'OK') else 'NG'

        df['Final_Status'] = df.apply(determine_status, axis=1)

        # --- NG理由ロジック (Reason) ---
        def determine_ng_reason(row):
            if row['Final_Status'] == 'OK':
                return 'OK'
            
            qr = row.get('QRresult', '').upper()
            if qr == 'NG': 
                return 'QR_NG'
            
            t_no = row.get('TestNo.', '')
            e_no = row.get('ErrorNo.', '')
            reason_str = f"{t_no} {e_no}".strip()
            
            if reason_str:
                return reason_str
            else:
                val = row.get('TESTresult', '')
                return val if val else 'Unknown_Error'

        df['NG_Reason'] = df.apply(determine_ng_reason, axis=1)
        
        return df, removed_count
    
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, 0

# --- 描画関数 ---

def plot_donut_chart(data, value_col='Final_Status', title='All Data'):
    counts = data[value_col].value_counts()
    if len(counts) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = {'OK': '#66b3ff', 'NG': '#ff9999'}
        labels = counts.index
        sizes = counts.values
        pie_colors = [colors.get(l, '#cccccc') for l in labels]
        
        def make_autopct(pct):
            total = sum(sizes)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val})'

        ax.pie(sizes, labels=labels, autopct=make_autopct,
               startangle=90, colors=pie_colors, textprops={'fontsize': 12},
               wedgeprops={'width': 0.5, 'edgecolor': 'white'}, pctdistance=0.85)
        ax.text(0, 0, f'{title}\nTotal: {sum(sizes)}', ha='center', va='center', fontsize=12, fontweight='bold')
        return fig
    return None

def plot_pareto_chart_fct_stacked(data, title='NG Reasons by FCT (Top 10 + Others)'):
    ng_df = data[data['NG_Reason'] != 'OK'].copy()
    
    if len(ng_df) == 0: return None
    total_ng_count = len(ng_df)
    ng_counts = ng_df['NG_Reason'].value_counts()
    
    top_reasons = ng_counts.head(10).index.tolist()
    ng_df.loc[~ng_df['NG_Reason'].isin(top_reasons), 'NG_Reason'] = 'Others'
    
    sort_order = top_reasons.copy()
    if 'Others' in ng_df['NG_Reason'].values: sort_order.append('Others')
    
    pivot_df = ng_df.pivot_table(index='NG_Reason', columns='FCT_ID', aggfunc='size', fill_value=0).reindex(sort_order)
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    
    # 積み上げ棒グラフ描画
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=cud_palette, edgecolor='white', linewidth=0.5)
    
    # 合計値の計算と表示
    total_per_reason = pivot_df.sum(axis=1)
    for i, v in enumerate(total_per_reason):
        ax1.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    ax1.set_ylabel('Count')
    ax1.set_title(f'{title}\n(NG Total: {total_ng_count})', fontsize=16)
    ax1.legend(title='FCT_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    cum_perc = (total_per_reason.cumsum() / total_ng_count) * 100
    
    ax2 = ax1.twinx()
    ax2.plot(pivot_df.index, cum_perc, color='black', marker='D', ms=5, linewidth=2)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    return fig

def plot_daily_trend_with_rate(data, title='Daily NG Trend & Defect Rate'):
    if 'Date' not in data.columns: return None
    unique_dates = sorted(data['Date'].dropna().unique())
    if not unique_dates: return None
    
    ng_data = data[data['Final_Status'] == 'NG'].copy()
    
    # NGデータの集計
    if len(ng_data) > 0:
        pivot_df = ng_data.pivot_table(index='Date', columns='FCT_ID', aggfunc='size', fill_value=0).reindex(unique_dates, fill_value=0)
    else:
        all_fcts = sorted(data['FCT_ID'].dropna().unique())
        if not all_fcts: return None
        pivot_df = pd.DataFrame(0, index=unique_dates, columns=all_fcts)

    # 検査総数と不良率の計算
    total_counts = data.groupby('Date').size().reindex(unique_dates, fill_value=0)
    daily_rate = (pivot_df.sum(axis=1) / total_counts * 100).fillna(0)
    daily_ng_total = pivot_df.sum(axis=1).values

    fig, ax1 = plt.subplots(figsize=(12, 6))
    cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    
    # 棒グラフ描画
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=cud_palette, edgecolor='white', width=0.8)
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('不良数 (件)')
    
    # 折れ線グラフ（不良率）描画
    ax2 = ax1.twinx()
    x_vals = np.arange(len(unique_dates))
    y_vals = daily_rate.values
    
    ax2.plot(x_vals, y_vals, color='red', linestyle='--', marker='o', linewidth=2, label='不良率')
    ax2.set_ylabel('不良率 (%)')
    
    # --- ラベル表示 ---
    counts_vals = total_counts.values
    
    for i in range(len(unique_dates)):
        total_val = counts_vals[i]
        ng_val = daily_ng_total[i]
        label_text = f"NG:{ng_val}\nTot:{total_val}"
        
        ax2.annotate(label_text, 
                     xy=(x_vals[i], y_vals[i]), 
                     xytext=(0, 10), 
                     textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    
    # Y軸の上限を調整
    if len(y_vals) > 0 and max(y_vals) > 0:
        ax2.set_ylim(0, max(y_vals) * 1.35)
    else:
        ax2.set_ylim(0, 10)
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    return fig

def render_grouped_pie_charts_grid(data, category_col, value_col='Final_Status', cols_per_row=4):
    if category_col not in data.columns: return
    unique_cats = sorted(data[category_col].unique())
    if not unique_cats: return
    
    colors = {'OK': '#66b3ff', 'NG': '#ff9999'}
    
    # グリッドの行ごとにループ
    for i in range(0, len(unique_cats), cols_per_row):
        cols = st.columns(cols_per_row)
        batch_cats = unique_cats[i:i + cols_per_row]
        
        for j, cat in enumerate(batch_cats):
            with cols[j]:
                subset = data[data[category_col] == cat]
                counts = subset[value_col].value_counts()
                total_sub = sum(counts.values)

                if total_sub > 0:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    
                    def make_autopct(pct):
                        val = int(round(pct * total_sub / 100.0))
                        return f'{pct:.1f}%\n({val})'

                    pie_colors = [colors.get(l, '#cccccc') for l in counts.index]
                    ax.pie(counts.values, labels=counts.index, autopct=make_autopct,
                           startangle=90, colors=pie_colors, wedgeprops={'width': 0.5})
                    ax.text(0, 0, f'{cat}\nTotal: {total_sub}', ha='center', va='center', fontweight='bold')
                    ax.set_title(f'{cat}', fontsize=10)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write(f"{cat}: No Data")

# --- メイン処理 ---

# 修正: サイドバーにファイルアップローダーを配置
st.sidebar.header("設定")
uploaded_file = st.sidebar.file_uploader("ログファイル(CSV)を選択", type=['csv'])

if uploaded_file is not None:
    # ファイルがアップロードされた場合のみ読み込みを実行
    df, removed_rows = load_data(uploaded_file)

    if df is not None:
        st.sidebar.header("表示フィルター")
        
        # --- データ範囲の自動取得 ---
        # データ内の日付範囲を取得（Noneを除去）
        valid_dates = df['Date'].dropna().sort_values()
        
        if not valid_dates.empty:
            # データが存在する場合はその最小・最大をデフォルトに
            default_min_date = valid_dates.iloc[0]
            default_max_date = valid_dates.iloc[-1]
        else:
            # データがない場合は今日の日付を仮定
            default_min_date = datetime.date.today()
            default_max_date = datetime.date.today()

        # --- 全期間ボタンの処理 ---
        def on_click_all_period():
            # ボタンが押されたらStateをデータの全期間にリセット
            st.session_state.filter_dates = (default_min_date, default_max_date)

        st.sidebar.button("全期間データを表示 (リセット)", on_click=on_click_all_period, type="primary")
        st.sidebar.markdown("---")

        # --- 日付範囲選択 (カレンダー) ---
        # 修正: valueには「デフォルト初期値」のみを渡します。
        selected_dates = st.sidebar.date_input(
            "期間選択 (Start - End)",
            value=(default_min_date, default_max_date), 
            min_value=default_min_date, 
            max_value=default_max_date,
            key='filter_dates'
        )

        # --- FCT_ID フィルター ---
        unique_fcts = sorted(df['FCT_ID'].dropna().unique())
        selected_fcts = []
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**FCT_ID フィルター**")
        
        # 省スペースのため2列で表示
        cols_fct = st.sidebar.columns(2)
        for i, fct in enumerate(unique_fcts):
            with cols_fct[i % 2]:
                if st.checkbox(fct, value=True, key=f"fct_{fct}"):
                    selected_fcts.append(fct)

        # --- フィルタリング処理 ---
        df_filtered = df.copy()
        
        # 1. 日付範囲フィルター
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_d, end_d = selected_dates
            
            mask = (df_filtered['Date'] >= start_d) & (df_filtered['Date'] <= end_d)
            df_filtered = df_filtered.loc[mask]
            
            period_info = f"{start_d} ～ {end_d}"
        else:
            if isinstance(selected_dates, tuple) and len(selected_dates) == 1:
                period_info = f"{selected_dates[0]} (終了日を選択してください)"
            else:
                period_info = "期間未定"
        
        # 2. FCTフィルター
        if selected_fcts:
            df_filtered = df_filtered[df_filtered['FCT_ID'].isin(selected_fcts)]
        else:
            df_filtered = df_filtered[0:0]
            st.sidebar.warning("FCT_ID が選択されていません。")

        st.info(f"ファイル: {uploaded_file.name} / 表示対象: {period_info} / 件数: {len(df_filtered)} 件")

        if len(df_filtered) > 0:
            
            # 1. 詳細分析
            st.subheader("詳細分析")
            tab1, tab2 = st.tabs(["FCT_ID別 判定結果", "Model別 判定結果"])
            
            with tab1:
                render_grouped_pie_charts_grid(df_filtered, 'FCT_ID', cols_per_row=3)
            with tab2:
                render_grouped_pie_charts_grid(df_filtered, 'Model', cols_per_row=3)
            
            st.markdown("---")

            # 2. 生産品質サマリー
            st.subheader("生産品質サマリー")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### OK/NG 比率")
                fig_all = plot_donut_chart(df_filtered, title="Selected Yield")
                if fig_all: 
                    st.pyplot(fig_all)
                    plt.close(fig_all)
                    
            with col2:
                st.markdown("#### NG内容 (パレート図)")
                fig_pareto = plot_pareto_chart_fct_stacked(df_filtered)
                if fig_pareto:
                    st.pyplot(fig_pareto)
                    plt.close(fig_pareto)
                else:
                    st.info("この期間・条件でのNGデータはありません。")
            
            # 3. 日別 不良数・不良率推移 (生産日のみ)
            st.markdown("#### 日別 不良数・不良率推移 (生産日のみ)")
            fig_trend = plot_daily_trend_with_rate(df_filtered)
            if fig_trend:
                st.pyplot(fig_trend)
                plt.close(fig_trend)
            else:
                 st.info("日別のデータはありません。")

            # 4. NG詳細履歴一覧 (全FCT合計)
            st.markdown("#### NG詳細履歴一覧 ")
            ng_only = df_filtered[df_filtered['Final_Status'] == 'NG'].copy()
            
            if not ng_only.empty:
                ng_only['TestErrorNo'] = ng_only['TestNo.'] + ' ' + ng_only['ErrorNo.']
                # --- 修正: nan nan を QR_NG に ---
                ng_only['TestErrorNo'] = ng_only['TestErrorNo'].replace('nan nan', 'QR_NG')
                
                display_cols = ['TestErrorNo', 'DateTime', 'PCB_Name', 'Model', 'FCT_ID', 'TESTresult']
                existing_cols = [c for c in display_cols if c in ng_only.columns]
                df_display = ng_only[existing_cols].sort_values(by='DateTime', ascending=False)
                
                # --- 交互色 (シマウマ模様) ---
                df_display = df_display.reset_index(drop=True)
                
                def alternate_color(row):
                    color = 'background-color: #e6f3ff' if row.name % 2 == 1 else ''
                    return [color] * len(row)
                
                st.markdown(f"NG件数: **{len(df_display)}** 件")
                
                st_df = df_display.style.apply(alternate_color, axis=1)
                
                # --- Streamlitでの表示 ---
                st.dataframe(st_df, hide_index=True, width='stretch')
                
            else:
                st.info("選択された条件でのNGデータはありません。")
        
        else:
            if selected_fcts: st.warning("選択された条件に一致するデータがありません。")
else:
    # ファイルがまだアップロードされていない時の初期表示
    st.info("サイドバーから分析したいCSVファイルをアップロードしてください。")