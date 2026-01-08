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


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import os

# ページ設定
st.set_page_config(page_title="Trace Log Analysis", layout="wide")

# タイトル
st.title("液晶演出生産時検査機OK/NGダッシュボード")

# CSVファイル設定
CSV_DIR = './CSV'
CSV_FILENAME = 'Trace_log_FCT4.csv'
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
        target_cols = ['Model', 'FCT_ID', 'QRresult', 'TESTresult', 'TestNo.', 'ErrorNo.']
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = ''
        
        # --- 日付情報の作成 ---
        if 'DateTime' in df.columns:
            df['DateTime_dt'] = pd.to_datetime(df['DateTime'])
            df['Month'] = df['DateTime_dt'].dt.strftime('%Y-%m')
            df['Date'] = df['DateTime_dt'].dt.date
        else:
            df['Month'] = 'Unknown'
            df['Date'] = None

        # --- 判定ロジック ---
        # 1. OK/NG判定
        def determine_status(row):
            qr = row.get('QRresult', 'NG')
            test = row.get('TESTresult', 'NG')
            if qr == 'OK' and test == 'OK':
                return 'OK'
            else:
                return 'NG'

        # 2. NG内容判定
        def determine_ng_reason(row):
            if row['QRresult'] == 'NG':
                return 'QR_NG'
            elif row['TESTresult'] == 'NG':
                return f"{row['TestNo.']} {row['ErrorNo.']}"
            return 'OK'

        df['Final_Status'] = df.apply(determine_status, axis=1)
        df['NG_Reason'] = df.apply(determine_ng_reason, axis=1)
        
        return df, removed_count
    
    except Exception as e:
        return None, 0

# 単一の円グラフ描画関数
def plot_donut_chart(data, value_col='Final_Status', title='All Data'):
    counts = data[value_col].value_counts()
    
    if len(counts) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        colors = {'OK': '#66b3ff', 'NG': '#ff9999'}
        labels = counts.index
        sizes = counts.values
        pie_colors = [colors.get(l, '#cccccc') for l in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100)})',
               startangle=90, colors=pie_colors, textprops={'fontsize': 12},
               wedgeprops={'width': 0.5, 'edgecolor': 'white'}, pctdistance=0.85)
        
        total = sum(sizes)
        center_text = f'{title}\nTotal: {total}'
        ax.text(0, 0, center_text, ha='center', va='center', fontsize=12, fontweight='bold')
        
        return fig
    else:
        return None

# パレート図描画関数 (FCT積み上げ版 + CUDカラー)
def plot_pareto_chart_fct_stacked(data, title='NG Reasons by FCT (Top 10 + Others)'):
    ng_df = data[data['NG_Reason'] != 'OK'].copy()
    
    if len(ng_df) == 0:
        return None
    
    total_ng_count = len(ng_df)
    
    # 1. 上位10件の特定
    ng_counts = ng_df['NG_Reason'].value_counts()
    top_n = 10
    top_reasons = ng_counts.head(top_n).index.tolist()
    
    # 2. その他への置換
    ng_df.loc[~ng_df['NG_Reason'].isin(top_reasons), 'NG_Reason'] = 'Others'
    
    # 3. ソート順
    sort_order = top_reasons.copy()
    if 'Others' in ng_df['NG_Reason'].values and 'Others' not in sort_order:
        sort_order.append('Others')
        
    # 4. ピボットテーブル作成
    pivot_df = ng_df.pivot_table(index='NG_Reason', columns='FCT_ID', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.reindex(sort_order)
    
    # 5. グラフ描画
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # CUDカラーパレット
    cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    num_fcts = len(pivot_df.columns)
    colors = [cud_palette[i % len(cud_palette)] for i in range(num_fcts)]
    
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=colors, edgecolor='white', linewidth=0.5)
    
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title}\n(NG Total: {total_ng_count})', fontsize=16)
    ax1.set_xlabel('NG Reason')
    ax1.legend(title='FCT_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. 累積比率線グラフ
    total_per_reason = pivot_df.sum(axis=1)
    cum_counts = total_per_reason.cumsum()
    cum_perc = (cum_counts / total_ng_count) * 100
    
    ax2 = ax1.twinx()
    ax2.plot(pivot_df.index, cum_perc, color='black', marker='D', ms=5, linewidth=2, label='Cumulative %')
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylabel('Cumulative %')
    ax2.set_ylim(0, 110)
    
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    fig.autofmt_xdate(rotation=45)
    
    plt.tight_layout()
    return fig

# --- 修正: 日別・FCT別積み上げグラフ + 不良率折れ線 ---
def plot_daily_trend_with_rate(data, title='Daily NG Trend & Defect Rate (Production Days Only)'):
    if 'Date' not in data.columns:
        return None
    
    # 1. フィルタリング後のデータから「生産があった日」のリストを取得
    unique_dates = sorted(data['Date'].dropna().unique())
    if not unique_dates:
        return None

    # 2. NGデータの集計 (FCT別)
    ng_data = data[data['Final_Status'] == 'NG'].copy()
    
    # ピボット作成 (行:日付, 列:FCT)
    if len(ng_data) > 0:
        pivot_df = ng_data.pivot_table(index='Date', columns='FCT_ID', aggfunc='size', fill_value=0)
        # 生産日すべてを含むように再インデックス (NGなしの日は0埋め)
        pivot_df = pivot_df.reindex(unique_dates, fill_value=0)
    else:
        # NGデータが全くない場合でも、生産日があればグラフ（空のバーと0%の線）を表示するために枠を作成
        all_fcts = sorted(data['FCT_ID'].dropna().unique())
        if not all_fcts: 
            return None # FCTもない場合は描画不可
        pivot_df = pd.DataFrame(0, index=unique_dates, columns=all_fcts)

    # 3. 日別総生産数 (OK + NG) の集計
    total_counts = data.groupby('Date').size().reindex(unique_dates, fill_value=0)
    
    # 4. 不良率の計算
    daily_ng_sum = pivot_df.sum(axis=1)
    daily_rate = (daily_ng_sum / total_counts) * 100
    daily_rate = daily_rate.fillna(0) # 0除算対策

    # 5. グラフ描画
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # CUDカラーパレット
    cud_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    num_fcts = len(pivot_df.columns)
    colors = [cud_palette[i % len(cud_palette)] for i in range(num_fcts)]
    
    # 積み上げ棒グラフ (NG数)
    # pandas plotはx軸を0, 1, 2...と配置する
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=colors, edgecolor='white', linewidth=0.5, width=0.8)
    
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('NG Count')
    ax1.set_xlabel('Date')
    ax1.legend(title='FCT_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. 折れ線グラフ (不良率) - 第2軸
    ax2 = ax1.twinx()
    # x軸の位置合わせ: 0, 1, 2...
    x_pos = np.arange(len(unique_dates))
    
    ax2.plot(x_pos, daily_rate.values, color='black', marker='o', markersize=4, linewidth=2, label='Defect Rate (%)')
    ax2.set_ylabel('Defect Rate (%)')
    ax2.set_ylim(bottom=0) # 0%からスタート
    
    # 折れ線グラフの凡例
    ax2.legend(loc='upper right')
    
    # 7. X軸ラベルの間引き処理
    n = len(unique_dates)
    if n > 20:
        step = n // 20 + 1
        for i, label in enumerate(ax1.get_xticklabels()):
            if i % step != 0:
                label.set_visible(False)
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

# グループ別円グラフ描画関数
def plot_grouped_pie_charts(data, category_col, value_col='Final_Status'):
    if category_col not in data.columns:
        return None

    unique_cats = sorted(data[category_col].unique())
    n_cats = len(unique_cats)
    
    if n_cats == 0:
        return None

    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6))
    if n_cats == 1:
        axes = [axes]
    
    colors = {'OK': '#66b3ff', 'NG': '#ff9999'}
    
    for ax, cat in zip(axes, unique_cats):
        subset = data[data[category_col] == cat]
        counts = subset[value_col].value_counts()
        
        if len(counts) > 0:
            labels = counts.index
            sizes = counts.values
            pie_colors = [colors.get(l, '#cccccc') for l in labels]
            
            ax.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p*sum(sizes)/100)})',
                   startangle=90, colors=pie_colors, textprops={'fontsize': 12},
                   wedgeprops={'width': 0.5, 'edgecolor': 'white'}, pctdistance=0.85)
            
            total = sum(sizes)
            center_text = f'{cat}\nTotal: {total}'
            ax.text(0, 0, center_text, ha='center', va='center', fontsize=12, fontweight='bold')
            
            ax.set_title(f'{category_col}: {cat}', fontsize=14, pad=20) 
            
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            ax.set_title(f'{category_col}: {cat}')
            ax.axis('off')

    plt.tight_layout()
    return fig

# --- メイン処理 ---
df, removed_rows = load_data(FILE_PATH)

if df is not None:
    # ---------------------------------------------------------
    # サイドバー設定
    # ---------------------------------------------------------
    st.sidebar.header("表示フィルター設定")

    # 1. 期間の切り替え
    unique_months = sorted(df['Month'].dropna().unique())
    period_options = ["全期間"] + list(unique_months)
    
    selected_period = st.sidebar.selectbox(
        "期間 選択",
        options=period_options,
        index=0
    )

    # 2. FCT_IDの切り替え (チェックボックス)
    st.sidebar.markdown("### FCT_ID 選択")
    unique_fcts = sorted(df['FCT_ID'].dropna().unique())
    
    selected_fcts = []
    for fct in unique_fcts:
        if st.sidebar.checkbox(fct, value=True, key=f"fct_{fct}"):
            selected_fcts.append(fct)

    # ---------------------------------------------------------
    # データのフィルタリング処理
    # ---------------------------------------------------------
    df_filtered = df.copy()

    # 期間
    if selected_period != "全期間":
        df_filtered = df_filtered[df_filtered['Month'] == selected_period]

    # FCT_ID
    if selected_fcts:
        df_filtered = df_filtered[df_filtered['FCT_ID'].isin(selected_fcts)]
    else:
        df_filtered = df_filtered[0:0]
        st.warning("FCT_ID が選択されていません。")

    # ---------------------------------------------------------
    # メイン画面表示
    # ---------------------------------------------------------
    st.info(f"表示対象: {selected_period} / FCT: {', '.join(selected_fcts) if selected_fcts else 'なし'} / データ件数: {len(df_filtered)} 件")

    if len(df_filtered) > 0:
        st.subheader("生産品質サマリー")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### OK/NG 比率")
            fig_all = plot_donut_chart(df_filtered, title="Selected Yield")
            if fig_all:
                st.pyplot(fig_all)
                
        with col2:
            st.markdown("#### NG内容 (パレート図)")
            fig_pareto = plot_pareto_chart_fct_stacked(df_filtered)
            if fig_pareto:
                st.pyplot(fig_pareto)
            else:
                st.info("この期間・条件でのNGデータはありません。")
        
        # --- 日別・FCT別 NG積み上げ + 不良率グラフ ---
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
            st.markdown("##### Model別")
            fig1 = plot_grouped_pie_charts(df_filtered, 'Model')
            if fig1:
                st.pyplot(fig1)
            else:
                 st.info("データがありません。")

        with tab2:
            st.markdown("##### FCT_ID別")
            fig2 = plot_grouped_pie_charts(df_filtered, 'FCT_ID')
            if fig2:
                st.pyplot(fig2)
            else:
                 st.info("データがありません。")
    
    else:
        if selected_fcts:
            st.warning("選択された条件に一致するデータがありません。")

else:
    st.error("エラー: ファイルが見つかりません。")