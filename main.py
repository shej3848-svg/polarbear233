# main.py
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 (깨짐 방지) ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

# ── CSV 읽기 (인코딩 이중 시도) ─────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ── 열 이름 표준화 (KeyError 방지) ──────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(
        df.columns
    ):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ── 같은 모양의 막대 그래프 재사용 함수 ────────────────────────────────
def make_bar_chart(df, col, title, color):
    valid = df.dropna(subset=[col])
    if len(valid) < 2:
        return None
    fig = px.bar(
        valid,
        x=valid.index.astype(str),
        y=col,
        labels={"x": "개체", "y": col},
        title=title,
        color=col,
        color_continuous_scale=color,
    )
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
    )
    return fig


# ── 사이드바: CSV 업로드 ────────────────────────────────────────────────
st.sidebar.title("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터 읽는 중..."):
        env_df = std_cols(read_csv(env_file))
if growth_file is not None:
    with st.spinner("생육 데이터 읽는 중..."):
        growth_df = std_cols(read_csv(growth_file))

# 학교명: 파일명 앞부분에서 추출
school_name = ""
if env_file is not None:
    school_name = env_file.name.split("_")[0]
elif growth_file is not None:
    school_name = growth_file.name.split("_")[0]

# 생육 측정 항목 선택
meas_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
selected_meas = st.sidebar.multiselect(
    "📏 생육 측정 항목 선택 (그래프에 표시)",
    options=meas_cols,
    default=meas_cols,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
if school_name:
    st.caption(f"📍 학교: {school_name}")

# ── 탭 구성 ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ═══════════════════════════════════════════════════════════════════════
# Tab 1 — 실험 개요
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            """
**극지식물**은 극한의 저온·건조 환경에서도 살아남는 식물로,  
같은 학교 실험실에서 키워도 개체마다 생육 차이가 꽤 크게 나타납니다.  

이 대시보드는 **우리 학교 극지식물 개체별 생육 차이**를  
- 평균·표준편차·변동계수(CV)로 요약하고,  
- 잎 수·지상부 길이와 생중량 사이에 어떤 관계가 있는지 상관분석하며,  
- 생중량 상위 25%와 하위 25% 개체를 비교해 어떤 차이가 있는지 살펴보는 도구입니다.  

> 💡 **통계 용어 간단 설명**  
> - **평균**: 모든 값을 더해 개수로 나눈 대표값  
> - **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 척도  
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 값 — 단위가 다른 집단끼리 비교할 때 유용  
> - **상관계수(r)**: 두 변수가 얼마나 함께 움직이는지 —1 ~ +1 사이 값, 0에 가까우면 관계 약함
"""
        )

        st.header("데이터 요약")

        # 요약 테이블
        summary_rows = []
        # 개체 수
        if len(growth_df) > 0:
            n_individuals = len(growth_df)
            summary_rows.append(["개체 수", f"{n_individuals}개체"])
        else:
            n_individuals = 0
            summary_rows.append(["개체 수", "업로드 필요"])

        # 측정 기간 (환경 데이터가 있으면)
        if "time" in env_df.columns and not env_df.empty:
            t_vals = env_df["time"].dropna()
            if len(t_vals) > 0:
                start = t_vals.min().strftime("%Y-%m-%d")
                end = t_vals.max().strftime("%Y-%m-%d")
                summary_rows.append(["측정 기간", f"{start} ~ {end}"])
            else:
                summary_rows.append(["측정 기간", "확인 불가"])
        else:
            summary_rows.append(["측정 기간", "환경 데이터 없음"])

        # 측정 항목
        meas_items = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
        if meas_items:
            summary_rows.append(["측정 항목", ", ".join(meas_items)])
        else:
            summary_rows.append(["측정 항목", "생육 데이터 없음"])

        if summary_rows:
            st.table(pd.DataFrame(summary_rows, columns=["구분", "값"]))
        else:
            st.info("데이터가 아직 업로드되지 않았습니다.")

        st.header("주요 지표 카드")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "개체번호" in growth_df.columns:
                st.metric("총 개체수", f"{len(growth_df)}개체")
            else:
                st.metric("총 개체수", "데이터 없음")

        with col2:
            if "temperature" in env_df.columns:
                tmp = env_df["temperature"].dropna()
                if len(tmp) >= 2:
                    st.metric("평균 온도", f"{tmp.mean():.1f}℃")
                else:
                    st.metric("평균 온도", "데이터 부족")
            else:
                st.metric("평균 온도", "환경 데이터 없음")

        with col3:
            if "humidity" in env_df.columns:
                hum = env_df["humidity"].dropna()
                if len(hum) >= 2:
                    st.metric("평균 습도", f"{hum.mean():.1f}%")
                else:
                    st.metric("평균 습도", "데이터 부족")
            else:
                st.metric("평균 습도", "환경 데이터 없음")

        with col4:
            if "생중량" in growth_df.columns:
                bw = growth_df["생중량"].dropna()
                if len(bw) >= 2:
                    st.metric("평균 생중량", f"{bw.mean():.2f}g")
                else:
                    st.metric("평균 생중량", "데이터 부족")
            else:
                st.metric("평균 생중량", "생중량 열 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════
# Tab 2 — 환경 데이터
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty or len(env_df) == 0:
            st.info("환경 데이터를 먼저 업로드해 주세요.")
        else:
            st.header("🌡️ 환경 요약 (평균·최소·최대)")

            env_stat_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_stat_cols:
                stats_rows = []
                for c in env_stat_cols:
                    vals = env_df[c].dropna()
                    if len(vals) >= 2:
                        stats_rows.append(
                            [c, f"{vals.mean():.2f}", f"{vals.min():.2f}", f"{vals.max():.2f}"]
                        )
                if stats_rows:
                    st.table(pd.DataFrame(stats_rows, columns=["항목", "평균", "최소", "최대"]))
                else:
                    st.info("환경 데이터 값이 충분하지 않아요.")
            else:
                st.info("환경 데이터에서 온도·습도·pH·EC 열을 찾을 수 없어요.")

            st.header("날짜별 시계열 (일별 평균)")

            # 시간 열이 있어야 시계열 가능
            if "time" in env_df.columns:
                daily = (
                    env_df.dropna(subset=["time"])
                    .copy()
                    .set_index("time")
                    .resample("D")
                    .mean()
                    .reset_index()
                )

                if not daily.empty:
                    # 온도
                    if "temperature" in daily.columns and daily["temperature"].notna().sum() >= 2:
                        fig_t = px.line(
                            daily,
                            x="time",
                            y="temperature",
                            labels={"time": "날짜", "temperature": "온도(℃)"},
                            title="일별 평균 온도 변화",
                        )
                        fig_t.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_t, width="stretch")

                    # 습도
                    if "humidity" in daily.columns and daily["humidity"].notna().sum() >= 2:
                        fig_h = px.line(
                            daily,
                            x="time",
                            y="humidity",
                            labels={"time": "날짜", "humidity": "습도(%)"},
                            title="일별 평균 습도 변화",
                        )
                        fig_h.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_h, width="stretch")

                    # pH·EC (열이 있을 때만)
                    ph_ec_cols = [c for c in ["ph", "ec"] if c in daily.columns and daily[c].notna().sum() >= 2]
                    if ph_ec_cols:
                        fig_sub = make_subplots(rows=1, cols=len(ph_ec_cols), subplot_titles=["pH"] if "ph" in ph_ec_cols else [] + ["EC"] if "ec" in ph_ec_cols else [])
                        col_idx = 1
                        if "ph" in ph_ec_cols:
                            fig_sub.add_trace(
                                go.Scatter(
                                    x=daily["time"], y=daily["ph"], name="pH", mode="lines"
                                ),
                                row=1,
                                col=col_idx,
                            )
                            col_idx += 1
                        if "ec" in ph_ec_cols:
                            fig_sub.add_trace(
                                go.Scatter(
                                    x=daily["time"], y=daily["ec"], name="EC", mode="lines"
                                ),
                                row=1,
                                col=col_idx,
                            )
                        fig_sub.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=300,
                        )
                        st.plotly_chart(fig_sub, width="stretch")
                    else:
                        st.info("pH·EC 열이 없거나 데이터가 충분하지 않아요.")
                else:
                    st.info("일별로 합산할 수 있는 데이터가 부족해요.")
            else:
                st.info("환경 데이터에 시간 열이 없어서 시계열을 그릴 수 없어요.")

            # expander: 원본 테이블 + CSV 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name="환경데이터_정리본.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════
# Tab 3 — 생육 결과
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty or len(growth_df) ==
