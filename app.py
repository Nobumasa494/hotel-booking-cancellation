import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np


st.set_page_config(
    page_title='Hotel Cancellation Predictor',
    page_icon='🎯',
    layout='wide'
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }
    h1 { color: #00d4ff; }
    h3 { color: #a0aec0; }
</style>
""", unsafe_allow_html=True)

model = joblib.load(r'C:\Users\nobu2\auto-data-analysis\hotel\model.pkl')
explainer = shap.TreeExplainer(model)

st.title('🎯 Hotel Cancellation Predictor')
st.markdown('##### キャンセルリスクをAIで予測し、ホテルの損失を最小化する')
st.markdown('---')

tab1, tab2 = st.tabs(['📋 個別予測', '📊 CSV一括分析'])

with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('### 予約情報を入力')
        lead_time = st.slider('📅 リードタイム（予約から到着までの日数）', 0, 500, 30)
        deposit = st.selectbox('💳 デポジット種別', ['No Deposit', 'Non Refund', 'Refundable'])
        special_requests = st.slider('⭐ 特別リクエスト数', 0, 5, 0)
        previous_cancellations = st.slider('⚠️ 過去のキャンセル回数', 0, 5, 0)
        adr = st.slider('💰 客室単価（$）', 0, 500, 100)
        is_repeated_guest = st.selectbox('🔄 リピーター', ['いいえ', 'はい'])

    with col_right:
        st.markdown('### 予測結果')
        if st.button('🔍 キャンセル確率を予測', use_container_width=True):
            input_data = pd.DataFrame({
                'lead_time': [lead_time],
                'arrival_date_year': [2024],
                'arrival_date_month': [6],
                'arrival_date_week_number': [24],
                'arrival_date_day_of_month': [15],
                'stays_in_weekend_nights': [1],
                'stays_in_week_nights': [2],
                'adults': [2],
                'children': [0],
                'babies': [0],
                'is_repeated_guest': [1 if is_repeated_guest == 'はい' else 0],
                'previous_cancellations': [previous_cancellations],
                'previous_bookings_not_canceled': [0],
                'booking_changes': [0],
                'agent': [0],
                'days_in_waiting_list': [0],
                'adr': [adr],
                'required_car_parking_spaces': [0],
                'total_of_special_requests': [special_requests],
                'prev_cancel_capped': [min(previous_cancellations, 5)],
                'hotel_Resort Hotel': [0],
                'meal_FB': [0], 'meal_HB': [0], 'meal_SC': [1], 'meal_Undefined': [0],
                'market_segment_Complementary': [0], 'market_segment_Corporate': [0],
                'market_segment_Direct': [1], 'market_segment_Groups': [0],
                'market_segment_Offline TA/TO': [0], 'market_segment_Online TA': [0],
                'market_segment_Undefined': [0],
                'distribution_channel_Direct': [1], 'distribution_channel_GDS': [0],
                'distribution_channel_TA/TO': [0], 'distribution_channel_Undefined': [0],
                'reserved_room_type_B': [0], 'reserved_room_type_C': [0],
                'reserved_room_type_D': [1], 'reserved_room_type_E': [0],
                'reserved_room_type_F': [0], 'reserved_room_type_G': [0],
                'reserved_room_type_H': [0], 'reserved_room_type_L': [0],
                'reserved_room_type_P': [0],
                'assigned_room_type_B': [0], 'assigned_room_type_C': [0],
                'assigned_room_type_D': [1], 'assigned_room_type_E': [0],
                'assigned_room_type_F': [0], 'assigned_room_type_G': [0],
                'assigned_room_type_H': [0], 'assigned_room_type_I': [0],
                'assigned_room_type_K': [0], 'assigned_room_type_L': [0],
                'assigned_room_type_P': [0],
                'deposit_type_Non Refund': [1 if deposit == 'Non Refund' else 0],
                'deposit_type_Refundable': [1 if deposit == 'Refundable' else 0],
                'customer_type_Group': [0], 'customer_type_Transient': [1],
                'customer_type_Transient-Party': [0],
                'country_BRA': [0], 'country_DEU': [0], 'country_ESP': [0],
                'country_FRA': [0], 'country_GBR': [0], 'country_IRL': [0],
                'country_ITA': [0], 'country_NLD': [0], 'country_Other': [0],
                'country_PRT': [0],
            })

            proba = model.predict_proba(input_data)[0][1]

            st.metric('キャンセル確率', f'{proba:.1%}')

            if proba >= 0.7:
                st.error('⚠️ 高リスク — デポジット要求またはリマインド連絡を推奨')
            elif proba >= 0.4:
                st.warning('🔔 注意 — チェックイン前に確認を推奨')
            else:
                st.success('✅ 低リスク — 通常対応でOK')

            fig, ax = plt.subplots(figsize=(5, 1))
            ax.barh([''], [proba], color='#ff4b4b' if proba >= 0.7 else '#ffa500' if proba >= 0.4 else '#00cc66', height=0.4)
            ax.barh([''], [1 - proba], left=[proba], color='#2d2d2d', height=0.4)
            ax.set_xlim(0, 1)
            ax.axis('off')
            st.pyplot(fig)

            st.markdown('### なぜこの確率？')
            shap_vals = explainer.shap_values(input_data)
            if isinstance(shap_vals, list):
                shap_cancel = shap_vals[1][0]
            elif shap_vals.ndim == 3:
                shap_cancel = shap_vals[0, :, 1]
            else:
                shap_cancel = shap_vals[:, 1]
            shap_series = pd.Series(shap_cancel, index=input_data.columns)
            top_reasons = shap_series.abs().sort_values(ascending=False).head(5).index

            for feat in top_reasons:
                val = shap_series[feat]
                direction = '⬆️ リスクを上げている' if val > 0 else '⬇️ リスクを下げている'
                st.write(f'**{feat}**: {direction} ({val:+.3f})')

with tab2:
    st.markdown('### CSVをアップロードして全予約を一括分析')
    st.caption('hotel_bookings.csv をそのままアップロードできます')

    uploaded = st.file_uploader('📂 CSVファイルをアップロード', type='csv')

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        df = df.drop(['reservation_status', 'reservation_status_date'], axis=1, errors='ignore')
        df['children'] = df['children'].fillna(0)
        df['country'] = df['country'].fillna('Unknown')
        df['agent'] = df['agent'].fillna(0)
        df = df.drop('company', axis=1, errors='ignore')
        df = df.drop('adr_bin', axis=1, errors='ignore')

        month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                     'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
        df['arrival_date_month'] = df['arrival_date_month'].map(month_map)
        df['prev_cancel_capped'] = df['previous_cancellations'].clip(upper=5)

        cat_cols = ['hotel','meal','market_segment','distribution_channel',
                    'reserved_room_type','assigned_room_type','deposit_type','customer_type']
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        top10 = ['PRT','GBR','FRA','ESP','DEU','ITA','IRL','BEL','BRA','NLD']
        df['country'] = df['country'].apply(lambda x: x if x in top10 else 'Other')
        df = pd.get_dummies(df, columns=['country'], drop_first=True)

        model_cols = model.feature_names_in_
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[model_cols]

        proba = model.predict_proba(df)[:, 1]
        result = pd.DataFrame({
            'キャンセル確率': (proba * 100).round(1).astype(str) + '%',
            'リスク': ['🔴 高リスク' if p >= 0.7 else '🟡 注意' if p >= 0.4 else '🟢 低リスク' for p in proba]
        })

        st.markdown('---')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('📋 総件数', f'{len(result):,}件')
        col2.metric('🔴 高リスク', f'{(proba >= 0.7).sum():,}件', f'{(proba >= 0.7).mean():.1%}')
        col3.metric('🟡 注意', f'{((proba >= 0.4) & (proba < 0.7)).sum():,}件')
        col4.metric('🟢 低リスク', f'{(proba < 0.4).sum():,}件')

        st.markdown('---')
        col_graph, col_table = st.columns([1, 2])

        with col_graph:
            st.markdown('### リスク分布')
            fig, ax = plt.subplots(figsize=(4, 4))
            sizes = [
                (proba >= 0.7).sum(),
                ((proba >= 0.4) & (proba < 0.7)).sum(),
                (proba < 0.4).sum()
            ]
            labels = ['高リスク', '注意', '低リスク']
            colors = ['#ff4b4b', '#ffa500', '#00cc66']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            st.pyplot(fig)

        with col_table:
            st.markdown('### 高リスク予約一覧')
            high_risk = result[result['リスク'] == '🔴 高リスク']
            st.write(f'{len(high_risk):,}件の高リスク予約')

            def highlight_risk(row):
                if '高リスク' in row['リスク']:
                    return ['background-color: #3d1515'] * len(row)
                elif '注意' in row['リスク']:
                    return ['background-color: #3d3015'] * len(row)
                return [''] * len(row)

            st.dataframe(result.style.apply(highlight_risk, axis=1), height=300)

        st.markdown('---')
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv = result.to_csv(index=False).encode('utf-8-sig')
            st.download_button('📥 全件ダウンロード', csv, 'cancel_prediction.csv', 'text/csv', use_container_width=True)
        with col_d2:
            csv_high = high_risk.to_csv(index=False).encode('utf-8-sig')
            st.download_button('🔴 高リスクのみダウンロード', csv_high, 'high_risk.csv', 'text/csv', use_container_width=True)
