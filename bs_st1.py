# -*- coding: utf-8 -*-
# Streamlitアプリケーション: Trace Log Analysis Dashboard (Strict OK Mode)
# 液晶演出生産時のTrace Logデータを解析し、判定結果を可視化します。
# QRresultとTESTresultの両方が'OK'の場合のみ'OK'と判定し、それ以外は'NG'とします。


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# ページ設定
st.set_page_config(page_title="Trace Log Analysis", layout="wide")

# タイトル
st.title("液晶演出生産時検査機OK/NGダッシュボード")

# CSVファイル設定
CSV_DIR = './CSV'
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
        target_cols = ['Model', 'FCT_ID', 'QRresult', 'TESTresult', 'TestNo.', 'ErrorNo.']
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = ''
        
        # --- 判定ロジック ---
        # 1. OK/NG判定 (Final_Status)
        def determine_status(row):
            qr = row.get('QRresult', 'NG')
            test = row.get('TESTresult', 'NG')
            if qr == 'OK' and test == 'OK':
                return 'OK'
            else:
                return 'NG'

        # 2. NG内容判定 (NG_Reason)
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

# 単一の円グラフ描画関数（ドーナツグラフ）
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
               wedgeprops={'width': 0.5, 'edgecolor': 'white'})
        
        ax.set_title(f'{title}\n(Total: {sum(sizes)})', fontsize=16)
        return fig
    else:
        return None

# パレート図描画関数 (Top 10 + Others)
def plot_pareto_chart(data, title='NG Reasons (Top 10 + Others)'):
    # NGデータのみ抽出
    ng_data = data[data['NG_Reason'] != 'OK']
    
    if len(ng_data) == 0:
        return None
    
    # 全NG項目のカウント（多い順）
    all_counts = ng_data['NG_Reason'].value_counts()
    total_ng_count = len(ng_data)
    
    if len(all_counts) > 0:
        # --- 変更箇所: 上位10件とそれ以外に分割 ---
        top_n = 10
        top_items = all_counts.iloc[:top_n]
        others_val = all_counts.iloc[top_n:].sum()
        
        # データフレームとして結合（Othersを最後に追加）
        plot_df = pd.DataFrame({'Count': top_items.values}, index=top_items.index)
        
        if others_val > 0:
            others_df = pd.DataFrame({'Count': [others_val]}, index=['Others'])
            plot_df = pd.concat([plot_df, others_df])
        
        # グラフ描画
        fig, ax1 = plt.subplots(figsize=(10, 6)) # 幅を少し広げました
        
        # 棒グラフ
        bars = ax1.bar(plot_df.index, plot_df['Count'], color='#ff9999')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{title}\n(NG Total: {total_ng_count})', fontsize=16)
        
        # 棒グラフの上に数値を表示
        ax1.bar_label(bars, label_type='edge', fontsize=10, padding=3)

        # 累積比率の計算
        cum_counts = plot_df['Count'].cumsum()
        cum_perc = (cum_counts / total_ng_count) * 100
        
        # 折れ線グラフ (累積比率)
        ax2 = ax1.twinx()
        ax2.plot(plot_df.index, cum_perc, color='#66b3ff', marker='D', ms=7, linewidth=2)
        
        # 軸の設定
        ax2.yaxis.set_major_formatter(PercentFormatter())
        ax2.set_ylabel('Cumulative %')
        ax2.set_ylim(0, 110)
        
        # グリッド線
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # x軸ラベルの回転
        fig.autofmt_xdate(rotation=45)
        
        return fig
    else:
        return None

# グループ別円グラフ描画関数
def plot_grouped_pie_charts(data, category_col, value_col='Final_Status'):
    if category_col not in data.columns:
        return None

    unique_cats = sorted(data[category_col].unique())
    n_cats = len(unique_cats)
    
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
                   wedgeprops={'width': 0.5, 'edgecolor': 'white'})
            
            ax.set_title(f'{category_col}: {cat}\n(Total: {sum(sizes)})', fontsize=16)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            ax.set_title(f'{category_col}: {cat}')
            ax.axis('off')

    plt.tight_layout()
    return fig

# --- メイン処理 ---
df, removed_rows = load_data(FILE_PATH)

if df is not None:
    st.success(f"データ読み込み完了: {len(df)} 件 (重複削除: {removed_rows} 件)")
    st.info("判定基準: QRresult='OK' かつ TESTresult='OK' → OK")

    st.subheader("生産品質サマリー")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### OK/NG 比率")
        fig_all = plot_donut_chart(df, title="Total Yield")
        if fig_all:
            st.pyplot(fig_all)
            
    with col2:
        st.markdown("#### NG内容 (パレート図)")
        fig_pareto = plot_pareto_chart(df)
        if fig_pareto:
            st.pyplot(fig_pareto)
        else:
            st.info("NGデータはありません。")

    st.markdown("---")

    st.subheader("Model別 判定結果")
    fig1 = plot_grouped_pie_charts(df, 'Model')
    if fig1:
        st.pyplot(fig1)

    st.markdown("---")

    st.subheader("FCT_ID別 判定結果")
    fig2 = plot_grouped_pie_charts(df, 'FCT_ID')
    if fig2:
        st.pyplot(fig2)

else:
    st.error("エラー: ファイルが見つかりません。")