import streamlit as st
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="스마트폰 중독 예측", layout="wide")

@st.cache_resource
def load_assets():
    model = load_model("model/addiction_model.h5")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")

model, scaler = load_assets()
df = load_data()

features = [
    'daily_screen_time_hours',
    'social_media_hours',
    'gaming_hours',
    'sleep_hours',
    'notifications_per_day',
    'app_opens_per_day'
]
target = 'addicted_label'

def classify_risk(prob):
    if prob < 0.25:
        return "정상"
    elif prob < 0.50:
        return "관심군"
    elif prob < 0.75:
        return "주의군"
    else:
        return "고위험군"

def get_interpretation(daily, sns, game, sleep, noti, app, risk):
    reasons = []
    if daily >= 8: reasons.append("일일 사용시간이 높음")
    if sns >= 3: reasons.append("SNS 사용시간이 많음")
    if game >= 2: reasons.append("게임 사용시간이 많음")
    if sleep <= 5.5: reasons.append("수면 시간이 부족함")
    if noti >= 120: reasons.append("알림 수가 많음")
    if app >= 70: reasons.append("앱 실행 횟수가 많음")

    base = ", ".join(reasons) if reasons else "특별히 과도한 패턴은 없음"

    if risk == "정상":
        return f"{base}. 전반적으로 안정적인 사용 패턴입니다."
    elif risk == "관심군":
        return f"{base}. 사용 습관을 점검할 필요가 있습니다."
    elif risk == "주의군":
        return f"{base}. 스마트폰 사용 패턴 관리가 필요합니다."
    else:
        return f"{base}. 과다 사용 경향이 강하여 개선이 필요합니다."

st.title("스마트폰 중독 위험도 예측 시스템")

tab1, tab2 = st.tabs(["사용자 예측", "실제값 vs 예측값"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        daily = st.slider("일일 사용시간", 0.0, 15.0, 5.0)
        sns = st.slider("SNS 사용시간", 0.0, 10.0, 2.0)
        game = st.slider("게임 사용시간", 0.0, 10.0, 1.0)

    with col2:
        sleep = st.slider("수면 시간", 0.0, 12.0, 6.0)
        noti = st.slider("알림 수", 0, 300, 80)
        app = st.slider("앱 실행 횟수", 0, 200, 40)

    st.subheader("입력값 요약")

    st.dataframe(pd.DataFrame({
        "항목": ["일일 사용시간","SNS","게임","수면","알림","앱실행"],
        "값": [daily,sns,game,sleep,noti,app]
    }), hide_index=True)

    st.subheader("위험도 기준")
    c1,c2,c3,c4 = st.columns(4)
    c1.success("정상\n0~0.24")
    c2.info("관심군\n0.25~0.49")
    c3.warning("주의군\n0.50~0.74")
    c4.error("고위험군\n0.75~1")

    if st.button("예측하기"):
        data = np.array([[daily,sns,game,sleep,noti,app]])
        data_scaled = scaler.transform(data)

        prob = float(model.predict(data_scaled)[0][0])
        risk = classify_risk(prob)

        st.subheader("예측 결과")

        st.metric("중독도", f"{prob*100:.2f}%")
        st.progress(prob)

        if risk == "정상":
            st.success(risk)
        elif risk == "관심군":
            st.info(risk)
        elif risk == "주의군":
            st.warning(risk)
        else:
            st.error(risk)

        st.markdown("### 결과 설명")
        st.write(
            f"중독도는 사용자 행동 패턴에 따른 중독의 정도를 의미합니다."
            f"현재 {prob*100:.2f}%로 {risk} 단계입니다."
        )

        st.write(get_interpretation(daily,sns,game,sleep,noti,app,risk))

with tab2:
    X = df[features]
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    compare_df = pd.DataFrame({
        "실제값": y_test.values,
        "예측값": y_pred,
        "확률": y_prob
    })

    compare_df["판정"] = np.where(compare_df["실제값"]==compare_df["예측값"],"정확","오분류")

    st.subheader("성능 요약")

    total = len(compare_df)
    correct = (compare_df["판정"]=="정확").sum()
    wrong = total - correct

    c1,c2,c3 = st.columns(3)
    c1.metric("전체", total)
    c2.metric("정확", correct)
    c3.metric("오분류", wrong)

    filter_opt = st.radio("보기",["전체","정확","오분류"],horizontal=True)

    if filter_opt=="정확":
        show_df = compare_df[compare_df["판정"]=="정확"]
    elif filter_opt=="오분류":
        show_df = compare_df[compare_df["판정"]=="오분류"]
    else:
        show_df = compare_df

    st.dataframe(show_df.head(20),use_container_width=True)

    st.download_button(
        "결과 다운로드",
        show_df.to_csv(index=False).encode("utf-8-sig"),
        "result.csv"
    )