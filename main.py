# main.py
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ---------- 한글 폰트 CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------- 캐시 데이터 로딩 ----------
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 CSV 읽기"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data
def std_cols(df):
    """열 이름 표준화 - 어떤 학교 CSV든 표준 이름으로 변환"""
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
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

def fig_layout(fig):
    """Plotly 한글 폰트 적용 (그래프 안 width 금지)"""
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig

def safe_metric(label, value, delta=None):
    """값이 있으면 st.metric, 없으면 st.info"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.info("📌 이 지표는 표시할 데이터가 없어요.")
    else:
        st.metric(label, value, delta=delta)

# ---------- 사이드바 ----------
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

env_file = st.sidebar.file_uploader("🌡️ 환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("📊 생육 결과 CSV 업로드", type="csv")

school_name = ""
if env_file is not None:
    school_name = env_file.name.split("_")[0].strip()
if growth_file is not None and not school_name:
    school_name = growth_file.name.split("_")[0].strip()

if school_name:
    st.sidebar.success(f"✅ 선택된 학교: **{school_name}**")
else:
    st.sidebar.info("📁 CSV 파일을 업로드하면 학교명이 자동으로 표시돼요.")

with st.sidebar.expander("📖 용어 설명"):
    st.write("- **평균**: 모든 값을 더해 개수로 나눈 대표값")
    st.write("- **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 값")
    st.write("- **CV(변동계수)**: 표준편차를 평균으로 나눈 값, 단위가 다른 집단 비교에 사용")
    st.write("- **상관계수(r)**: 두 변수가 함께 움직이는 정도(-1~1), 1에 가까울수록 강한 양의 관계")

# ---------- 데이터 로딩 (한 번만) ----------
env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("🌡️ 환경 데이터를 읽는 중..."):
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)
if growth_file is not None:
    with st.spinner("📊 생육 데이터를 읽는 중..."):
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)

# ---------- 탭 ----------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ===== Tab 1: 실험 개요 =====
with tab1:
    try:
        if env_df.empty and growth_df.empty:
            st.info("📁 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
        else:
            # 연구 배경 및 목적
            st.header("연구 배경 및 목적")
            st.write("""
**극지식물**은 추운 환경에서 자라는 식물로, 같은 장소에서도 개체마다 생육 상태가 다를 수 있어요.
이 대시보드에서는 우리 학교 극지식물 개체들의 **생중량(몸무게)** 분포를 확인하고,
잎 수·지상부 길이와 생중량의 관계를 분석해요.
""")
            st.write("**분석 항목**")
            st.write("- 개체별 생중량 분포와 차이 (평균·표준편차·CV)")
            st.write("- 잎 수·지상부 길이와 생중량의 상관관계")
            st.write("- 생중량 상위 25% vs 하위 25% 개체 비교")

            # 데이터 요약 표
            st.subheader("데이터 요약")
            summary_rows = []
            if not env_df.empty:
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                env_count = len(env_cols)
                time_min = env_df["time"].min() if "time" in env_df.columns else None
                time_max = env_df["time"].max() if "time" in env_df.columns else None
                summary_rows.append({
                    "데이터": "환경 데이터",
                    "행 수": len(env_df),
                    "측정 항목 수": env_count,
                    "측정 기간": f"{time_min.strftime('%Y-%m-%d') if time_min is not None else 'N/A'} ~ {time_max.strftime('%Y-%m-%d') if time_max is not None else 'N/A'}"
                })
            if not growth_df.empty:
                growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                summary_rows.append({
                    "데이터": "생육 결과 데이터",
                    "행 수": len(growth_df),
                    "측정 항목 수": len(growth_cols),
                    "측정 기간": "개체별 1회 측정"
                })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), width="stretch")
            else:
                st.info("📌 표시할 데이터 요약이 없어요.")

            # 주요 지표 카드 4개
            st.subheader("주요 지표")
            col1, col2, col3, col4 = st.columns(4)

            # 총 개체수
            with col1:
                if not growth_df.empty:
                    safe_metric("총 개체수", len(growth_df))
                else:
                    st.info("📌 개체수 데이터가 없어요.")

            # 평균 온도
            with col2:
                if not env_df.empty and "temperature" in env_df.columns:
                    t = env_df["temperature"].dropna()
                    if len(t) >= 2:
                        safe_metric("평균 온도 (°C)", round(t.mean(), 1))
                    else:
                        st.info("📌 온도 데이터가 부족해요.")
                else:
                    st.info("📌 온도 데이터가 없어요.")

            # 평균 습도
            with col3:
                if not env_df.empty and "humidity" in env_df.columns:
                    h = env_df["humidity"].dropna()
                    if len(h) >= 2:
                        safe_metric("평균 습도 (%)", round(h.mean(), 1))
                    else:
                        st.info("📌 습도 데이터가 부족해요.")
                else:
                    st.info("📌 습도 데이터가 없어요.")

            # 평균 생중량
            with col4:
                if not growth_df.empty and "생중량" in growth_df.columns:
                    gw = growth_df["생중량"].dropna()
                    if len(gw) >= 2:
                        safe_metric("평균 생중량 (g)", round(gw.mean(), 2))
                    else:
                        st.info("📌 생중량 데이터가 부족해요.")
                else:
                    st.info("📌 생중량 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===== Tab 2: 환경 데이터 =====
with tab2:
    try:
        if env_df.empty:
            st.info("📁 왼쪽 사이드바에서 환경 CSV를 업로드해 주세요.")
        else:
            st.header("🌡️ 환경 데이터 분석")

            # 환경 요약 표 (평균·최소·최대)
            st.subheader("환경 요약")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                summary_data = []
                for c in env_cols:
                    vals = env_df[c].dropna()
                    if len(vals) >= 2:
                        summary_data.append({
                            "항목": c,
                            "평균": round(vals.mean(), 2),
                            "최소": round(vals.min(), 2),
                            "최대": round(vals.max(), 2)
                        })
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), width="stretch")
                else:
                    st.info("📌 요약할 환경 데이터가 부족해요.")
            else:
                st.info("📌 환경 데이터에서 인식할 수 있는 열이 없어요.")

            # 날짜별 시계열 (일별 평균)
            st.subheader("날짜별 환경 변화 (일별 평균)")
            if "time" in env_df.columns:
                env_df_daily = env_df.copy()
                env_df_daily["date"] = env_df_daily["time"].dt.date
                daily = env_df_daily.groupby("date").mean(numeric_only=True)

                # 온도 변화
                if "temperature" in daily.columns:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=daily.index,
                        y=daily["temperature"],
                        mode="lines",
                        name="온도 (°C)",
                        line=dict(color="#e74c3c", width=2)
                    ))
                    fig_temp = fig_layout(fig_temp)
                    fig_temp.update_layout(
                        title="일별 평균 온도 변화",
                        xaxis_title="날짜",
                        yaxis_title="온도 (°C)",
                        height=300
                    )
                    st.plotly_chart(fig_temp, width="stretch")

                # 습도 변화
                if "humidity" in daily.columns:
                    fig_hum = go.Figure()
                    fig_hum.add_trace(go.Scatter(
                        x=daily.index,
                        y=daily["humidity"],
                        mode="lines",
                        name="습도 (%)",
                        line=dict(color="#3498db", width=2)
                    ))
                    fig_hum = fig_layout(fig_hum)
                    fig_hum.update_layout(
                        title="일별 평균 습도 변화",
                        xaxis_title="날짜",
                        yaxis_title="습도 (%)",
                        height=300
                    )
                    st.plotly_chart(fig_hum, width="stretch")

                # pH·EC 변화 (열이 있을 때만)
                ph_ec_cols = [c for c in ["ph", "ec"] if c in daily.columns]
                if ph_ec_cols:
                    fig_ph_ec = make_subplots(rows=len(ph_ec_cols), cols=1,
                                              subplot_titles=["pH 변화", "EC 변화"] if len(ph_ec_cols) == 2 else [f"{ph_ec_cols[0].upper()} 변화"])
                    colors = {"ph": "#2ecc71", "ec": "#9b59b6"}
                    for i, c in enumerate(ph_ec_cols):
                        fig_ph_ec.add_trace(go.Scatter(
                            x=daily.index,
                            y=daily[c],
                            mode="lines",
                            name=c.upper(),
                            line=dict(color=colors.get(c, "#333"), width=2)
                        ), row=i+1, col=1)
                    fig_ph_ec = fig_layout(fig_ph_ec)
                    fig_ph_ec.update_layout(
                        title="일별 pH·EC 변화",
                        height=250 * len(ph_ec_cols),
                        showlegend=False
                    )
                    st.plotly_chart(fig_ph_ec, width="stretch")
                else:
                    st.info("📌 pH·EC 데이터가 없어서 이 그래프는 건너뛰었어요.")
            else:
                st.info("📌 시간(time) 열이 없어서 일별 그래프를 만들 수 없어요.")

            # 원본 테이블 + CSV 다운로드
            with st.expander("📄 환경 데이터 원본 테이블"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ 환경 CSV 다운로드 (utf-8-sig)",
                        data=csv_env,
                        file_name=f"{school_name}_환경데이터.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("📌 표시할 환경 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ===== Tab 3: 생육 결과 =====
with tab3:
    try:
        if growth_df.empty:
            st.info("📁 왼쪽 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.header("📊 생육 결과 분석")

            # 🥇 핵심 결과 카드
            st.subheader("🥇 핵심 결과")
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("평균 생중량 (g)", round(gw.mean(), 2))
                    with c3:
                        safe_metric("표준편차 (g)", round(gw.std(), 2))
                    with c2:
                        # 최대·최소 개체
                        if len(gw) >= 1:
                            max_idx = gw.idxmax()
                            min_idx = gw.idxmin()
                            max_val = gw.max()
                            min_val = gw.min()
                            max_id = growth_df.loc[max_idx, "개체번호"] if "개체번호" in growth_df.columns else "?"
                            min_id = growth_df.loc[min_idx, "개체번호"] if "개체번호" in growth_df.columns else "?"
                            st.write(f"🔺 최대: 개체 **{max_id}** — {round(max_val, 2)} g")
                            st.write(f"🔻 최소: 개체 **{min_id}** — {round(min_val, 2)} g")
                        else:
                            st.info("📌 생중량 데이터가 부족해요.")
                else:
                    st.info("📌 생중량 데이터가 부족해요.")

            # 개체별 생육 비교 (막대 그래프)
            st.subheader("개체별 생육 비교")
            growth_cols_bar = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"] if c in growth_df.columns]
            if growth_cols_bar:
                sel_col = st.selectbox(
                    "📌 비교할 측정 항목 선택",
                    growth_cols_bar,
                    index=0
                )
                vals = growth_df[sel_col].dropna()
                ids = growth_df["개체번호"] if "개체번호" in growth_df.columns else range(len(growth_df))
                # 생중량이면 상위·하위 25% 강조
                if sel_col == "생중량" and len(vals) >= 4:
                    q75 = vals.quantile(0.75)
                    q25 = vals.quantile(0.25)
                else:
                    q75 = q25 = None

                fig_bar = go.Figure()
                colors_bar = []
                for i, (idx, row) in enumerate(growth_df.iterrows()):
                    v = row[sel_col]
                    if pd.isna(v):
                        continue
                    if q75 is not None and v >= q75:
                        colors_bar.append("#e74c3c")  # 상위 25% 빨간색
                    elif q25 is not None and v <= q25:
                        colors_bar.append("#3498db")  # 하위 25% 파란색
                    else:
                        colors_bar.append("#2ecc71")  # 중간 초록색
                    fig_bar.add_trace(go.Bar(
                        x=[row["개체번호"] if "개체번호" in growth_df.columns else i],
                        y=[v],
                        name=str(row["개체번호"] if "개체번호" in growth_df.columns else i),
                        marker_color=colors_bar[-1],
                        showlegend=False
                    ))
                fig_bar = fig_layout(fig_bar)
                fig_bar.update_layout(
                    title=f"개체별 {sel_col} 비교 (⭐ 빨강: 상위 25%, 파랑: 하위 25%)",
                    xaxis_title="개체번호",
                    yaxis_title=sel_col,
                    height=400,
                    bargap=0.15
                )
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("📌 생육 데이터에서 인식할 수 있는 측정 열이 없어요.")

            # 생중량 분포 (박스플롯)
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=gw,
                        name="생중량 (g)",
                        fillcolor="#2ecc71",
                        boxpoints="outliers",
                        marker=dict(color="#e74c3c", size=6),
                        line=dict(color="#27ae60")
                    ))
                    fig_box = fig_layout(fig_box)
                    fig_box.update_layout(
                        title="생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        height=300
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    # CV (변동계수) 표시
                    mean_gw = gw.mean()
                    std_gw = gw.std()
                    if mean_gw > 0:
                        cv = (std_gw / mean_gw) * 100
                        st.info(f"📌 **CV(변동계수)**: {round(cv, 1)}% — 값이 클수록 개체 간 차이가 커요.")
                else:
                    st.info("📌 생중량 데이터가 부족해서 박스플롯을 만들 수 없어요.")
            else:
                st.info("📌 생중량 열이 없어서 분포 그래프를 만들 수 없어요.")

            # 상관관계 분석 (산점도 2개 + 추세선)
            st.subheader("상관관계 분석")
            corr_pairs = [
                ("잎 수", "생중량"),
                ("지상부 길이", "생중량")
            ]
            for x_col, y_col in corr_pairs:
                if x_col in growth_df.columns and y_col in growth_df.columns:
                    df_c = growth_df[[x_col, y_col]].dropna()
                    if len(df_c) >= 2:
                        x = df_c[x_col].values
                        y = df_c[y_col].values
                        r = np.corrcoef(x, y)[0, 1]

                        # 추세선 (np.polyfit)
                        slope, intercept = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = slope * x_line + intercept

                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            name="데이터",
                            marker=dict(size=10, color="#3498db", line=dict(width=1, color="#2980b9"))
                        ))
                        fig_scatter.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name=f"추세선 (r={round(r, 3)})",
                            line=dict(color="#e74c3c", width=2, dash="dash")
                        ))
                        fig_scatter = fig_layout(fig_scatter)
                        fig_scatter.update_layout(
                            title=f"{x_col} vs {y_col} (상관계수 r = {round(r, 3)})",
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            height=350,
                            legend=dict(x=0.02, y=0.98)
                        )
                        st.plotly_chart(fig_scatter, width="stretch")

                        # 상관계수 해석 안내
                        abs_r = abs(r)
                        if abs_r >= 0.7:
                            strength = "강한"
                        elif abs_r >= 0.4:
                            strength = "중간 정도"
                        else:
                            strength = "약한"
                        direction = "양의" if r > 0 else "음의"
                        st.info(f"📌 **{x_col}과 {y_col}의 관계**: {strength} {direction} 상관관계 (r = {round(r, 3)})")
                    else:
                        st.info(f"📌 {x_col}과 {y_col} 데이터가 부족해서 상관관계를 계산할 수 없어요.")
                else:
                    missing = [c for c in [x_col, y_col] if c not in growth_df.columns]
                    st.info(f"📌 {', '.join(missing)} 열이 없어서 이 상관분석은 건너뛰었어요.")

            # 원본 테이블 + CSV 다운로드
            with st.expander("📄 생육 데이터 원본 테이블"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    csv_growth = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ 생육 CSV 다운로드 (utf-8-sig)",
                        data=csv_growth,
                        file_name=f"{school_name}_생육결과데이터.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("📌 표시할 생육 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
