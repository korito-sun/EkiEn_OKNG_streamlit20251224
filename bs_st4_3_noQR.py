# -*- coding: utf-8 -*-
# Streamlitアプリケーション: Trace Log Analysis Dashboard (Strict OK Mode)
# 液晶演出生産時のTrace Logデータを解析し、判定結果を可視化します。
# QRresultとTESTresultの両方が'OK'の場合のみ'OK'と判定し、それ以外は'NG'とします。
# また、日別の不良数・不良率推移グラフを生産日のみ表示するように修正しています。
# さらに、NG理由のパレート図はFCT別積み上げ表示とし、CUDカラーパレットを使用しています。
# また、グループ別円グラフ描画関数を追加し、Model別・FCT_ID別のOK/NG比率を表示できるようにしています。
# 2025-12-19 改訂版 QRコードがない場合のダッシュボードプロト版
# QRコードがない場合は、OK判定は存在しないものとし、全てNG扱いとします。→OK/NG判定ロジックを修正必要。
# QRコードがない場合は、"QRresult"列をOK扱いとし、TESTresultの結果のみでOK/NG判定を行います。

# 2025-12-25 改訂版 NG理由の詳細表を追加
# パレート図の下に、NG理由の詳細な統計テーブルを追加しました。
# NG詳細表のカラムを特定項目（DateTime, PCB_Name等）に指定し、横長に表示します。
# 1. col1(円グラフ)とcol2(パレート図)の下に、NG詳細履歴一覧表を追加配置。
# 2. 表の項目は TestNo., ErrorNo., DateTime, PCB_Name, Model, FCT_ID の順。
# 3. データ読み込み時に PCB_Name カラムもクリーニング対象に追加。
# 修正内容: NG詳細履歴表の 'TestNo.' と 'ErrorNo.' を結合し、'TestErrorNo' として表示。
# 修正内容: 
# Deprecation Warning対応: st.dataframeの use_container_width=True を width='stretch' に変更しました。

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import os
import matplotlib

# 日本語フォント設定　#2026-01-10 修正
matplotlib.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ページ設定
st.set_page_config(page_title="Trace Log Analysis", layout="wide")

# タイトル
st.title("液晶演出生産時検査機OK/NGダッシュボード")

# CSVファイル設定
CSV_DIR = './CSV'
# CSV_FILENAME = 'Trace_log_FCT4.csv'
CSV_FILENAME = '20251201A84_Trace_log.csv'
FILE_PATH = os.path.join(CSV_DIR, CSV_FILENAME)

# データ読み込みとキャッシュ
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        
        # 重複データの削除
        initial_count = len(df)
        df = df.drop_duplicates()
        dedup_count = len(df)
        removed_count = initial_count - dedup_count
        
        # クリーニング
        target_cols = ['Model', 'FCT_ID', 'QRresult', 'TESTresult', 'TestNo.', 'ErrorNo.', 'PCB_Name', 'DateTime']
        for col in target_cols:
            if col in df.columns:
                # 文字列化して空白除去
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = ''
        
        # --- 日付情報の作成 ---
        if 'DateTime' in df.columns:
            df['DateTime_dt'] = pd.to_datetime(df['DateTime'], errors='coerce')
            df['Month'] = df['DateTime_dt'].dt.strftime('%Y-%m')
            df['Date'] = df['DateTime_dt'].dt.date
        else:
            df['Month'] = 'Unknown'
            df['Date'] = None

        # --- 判定ロジック ---
        def determine_status(row):
            qr = row.get('QRresult', '').upper()
            test = row.get('TESTresult', 'NG').upper()
            is_qr_ok = (qr == 'OK' or qr in ['', 'NAN', 'NONE'])
            return 'OK' if (is_qr_ok and test == 'OK') else 'NG'

        def determine_ng_reason(row):
            qr = row.get('QRresult', '').upper()
            test = row.get('TESTresult', 'NG').upper()
            if qr == 'NG': return 'QR_NG'
            elif test == 'NG': return f"{row['TestNo.']} {row['ErrorNo.']}"
            return 'OK'

        df['Final_Status'] = df.apply(determine_status, axis=1)
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
        ax.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100)})',
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
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=cud_palette, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title}\n(NG Total: {total_ng_count})', fontsize=16)
    ax1.legend(title='FCT_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    total_per_reason = pivot_df.sum(axis=1)
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
    
    if len(ng_data) > 0:
        pivot_df = ng_data.pivot_table(index='Date', columns='FCT_ID', aggfunc='size', fill_value=0).reindex(unique_dates, fill_value=0)
    else:
        all_fcts = sorted(data['FCT_ID'].dropna().unique())
        if not all_fcts: return None
        pivot_df = pd.DataFrame(0, index=unique_dates, columns=all_fcts)

    total_counts = data.groupby('Date').size().reindex(unique_dates, fill_value=0)
    daily_rate = (pivot_df.sum(axis=1) / total_counts * 100).fillna(0)
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=cud_palette, edgecolor='white', width=0.8)
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('不良数 (件)') #2026-01-10 修正
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(unique_dates)), daily_rate.values, color='black', marker='o', linewidth=2)
    ax2.set_ylabel('不良率 (%)') #2026-01-10 修正
    ax2.set_ylim(bottom=0)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_grouped_pie_charts(data, category_col, value_col='Final_Status'):
    if category_col not in data.columns: return None
    unique_cats = sorted(data[category_col].unique())
    if not unique_cats: return None
    fig, axes = plt.subplots(1, len(unique_cats), figsize=(6 * len(unique_cats), 6))
    if len(unique_cats) == 1: axes = [axes]
    colors = {'OK': '#66b3ff', 'NG': '#ff9999'}
    for ax, cat in zip(axes, unique_cats):
        subset = data[data[category_col] == cat]
        counts = subset[value_col].value_counts()
        if len(counts) > 0:
            ax.pie(counts.values, labels=counts.index, autopct=lambda p: f'{p:.1f}%\n({int(p*sum(counts.values)/100)})',
                   startangle=90, colors=[colors.get(l, '#cccccc') for l in counts.index], wedgeprops={'width': 0.5})
            ax.text(0, 0, f'{cat}\nTotal: {sum(counts.values)}', ha='center', va='center', fontweight='bold')
            ax.set_title(f'{category_col}: {cat}')
        else:
            ax.axis('off')
    plt.tight_layout()
    return fig

# --- メイン処理 ---
df, removed_rows = load_data(FILE_PATH)

if df is not None:
    st.sidebar.header("表示フィルター設定")
    unique_months = sorted(df['Month'].dropna().unique())
    selected_period = st.sidebar.selectbox("期間 選択", options=["全期間"] + list(unique_months))
    unique_fcts = sorted(df['FCT_ID'].dropna().unique())
    selected_fcts = [fct for fct in unique_fcts if st.sidebar.checkbox(fct, value=True, key=f"fct_{fct}")]

    df_filtered = df.copy()
    if selected_period != "全期間":
        df_filtered = df_filtered[df_filtered['Month'] == selected_period]
    if selected_fcts:
        df_filtered = df_filtered[df_filtered['FCT_ID'].isin(selected_fcts)]
    else:
        df_filtered = df_filtered[0:0]
        st.warning("FCT_ID が選択されていません。")

    st.info(f"表示対象: {selected_period} / データ件数: {len(df_filtered)} 件")

    if len(df_filtered) > 0:
        st.subheader("生産品質サマリー")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### OK/NG 比率")
            fig_all = plot_donut_chart(df_filtered, title="Selected Yield")
            if fig_all: st.pyplot(fig_all)
                
        with col2:
            st.markdown("#### NG内容 (パレート図)")
            fig_pareto = plot_pareto_chart_fct_stacked(df_filtered)
            if fig_pareto:
                st.pyplot(fig_pareto)
            else:
                st.info("この期間・条件でのNGデータはありません。")
        
        # -------------------------------------------------------------------
        # NG詳細履歴一覧 (修正済み: width='stretch')
        # -------------------------------------------------------------------
        st.markdown("#### NG詳細履歴一覧 (全FCT合計)")
        ng_only = df_filtered[df_filtered['Final_Status'] == 'NG'].copy()
        
        if not ng_only.empty:
            # カラムの結合処理: TestNo.とErrorNo.を結合
            ng_only['TestErrorNo'] = ng_only['TestNo.'] + ' ' + ng_only['ErrorNo.']
            
            # 表示カラム順
            display_cols = ['TestErrorNo', 'DateTime', 'PCB_Name', 'Model', 'FCT_ID']
            
            # カラムフィルタ
            existing_cols = [c for c in display_cols if c in ng_only.columns]
            
            # ソート
            df_display = ng_only[existing_cols].sort_values(by='DateTime', ascending=False)
            
            # 表を表示 (Warning対応: use_container_width -> width='stretch')
            st.dataframe(df_display, width='stretch', hide_index=True)
        else:
            st.info("選択された条件でのNGデータはありません。")
        # -------------------------------------------------------------------

        st.markdown("#### 日別 不良数・不良率推移 (生産日のみ)")
        fig_trend = plot_daily_trend_with_rate(df_filtered)
        if fig_trend:
            st.pyplot(fig_trend)
        else:
             st.info("日別のデータはありません。")

        st.markdown("---")
        st.subheader("詳細分析")
        tab1, tab2 = st.tabs(["Model別 判定結果", "FCT_ID別 判定結果"])
        with tab1:
            fig1 = plot_grouped_pie_charts(df_filtered, 'Model')
            if fig1: st.pyplot(fig1)
        with tab2:
            fig2 = plot_grouped_pie_charts(df_filtered, 'FCT_ID')
            if fig2: st.pyplot(fig2)
    
    else:
        if selected_fcts: st.warning("選択された条件に一致するデータがありません。")

else:
    st.error("エラー: ファイルが見つかりません。")