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

def insight_box(title, message, kind="info"):
    """데이터 특성에 따른 인사이트를 색상 박스로 표시"""
    icon = {"info": "📌", "good": "✅", "warn": "⚠️", "note": "💡"}.get(kind, "📌")
    color = {
        "info": "#e8f4f8",
        "good": "#e8f5e9",
        "warn": "#fff3e0",
        "note": "#f3e5f5"
    }.get(kind, "#e8f4f8")
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:12px 16px; border-radius:8px;
                    border-left:4px solid {'#3498db' if kind=='info' else '#27ae60' if kind=='good' else '#e67e22' if kind=='warn' else '#8e44ad'};
                    margin-bottom:12px;">
            <b>{icon} {title}</b><br>
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )

def describe_distribution(series, name, unit=""):
    """데이터 분포 특성에 따른 설명 생성"""
    s = series.dropna()
    if len(s) < 2:
        return "📌 데이터가 부족해서 분포를 분석할 수 없어요."
    mean_v = s.mean()
    std_v = s.std()
    cv = (std_v / mean_v * 100) if mean_v > 0 else 0
    min_v = s.min()
    max_v = s.max()
    rng = max_v - min_v

    parts = []
    # 분포 형태 (왜도)
    skew = s.skew()
    if abs(skew) < 0.5:
        shape = "대체로 대칭적인 분포"
    elif skew > 0.5:
        shape = "오른쪽(큰 값)으로 꼬리가 긴 치우친 분포"
    else:
        shape = "왼쪽(작은 값)으로 꼬리가 긴 치우친 분포"

    parts.append(f"**분포 형태**: {shape}")

    # CV에 따른 개체 간 차이 해석
    if cv < 20:
        parts.append(f"**개체 간 차이가 비교적 작아요** (CV {round(cv, 1)}%). 대부분의 개체가 비슷한 수준으로 자랐어요.")
    elif cv < 50:
        parts.append(f"**개체 간 차이가 중간 정도**예요 (CV {round(cv, 1)}%). 환경에 따라 생육 차이가 나타나기 시작해요.")
    else:
        parts.append(f"**개체 간 차이가 매우 커요** (CV {round(cv, 1)}%). 잘 자란 개체와 그렇지 않은 개체가 뚜렷하게 구분돼요.")

    # 범위 설명
    parts.append(f"**범위**: 최소 {round(min_v, 2)}{unit} ~ 최대 {round(max_v, 2)}{unit} (차이 {round(rng, 2)}{unit})")

    # 이상치 개수
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((s < lower) | (s > upper)).sum()
    if outlier_count > 0:
        outlier_ids = []
        if "개체번호" in s.index.get_level_values(0).name if isinstance(s.index, pd.MultiIndex) else "개체번호" in s.axes[0].name if hasattr(s, 'axes') else False:
            pass
        parts.append(f"**이상치**: {outlier_count}개체가 일반적인 범위에서 벗어나 있어요 (IQR 기준).")

    return "<br>".join(parts)

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
    st.write("- **CV(변동계수)**: 표준편차를 평균으로 나눈 값(%) — 단위가 다른 집단 비교에 사용, 값이 클수록 차이가 큼")
    st.write("- **상관계수(r)**: 두 변수가 함께 움직이는 정도(-1~1), 0에 가까우면 관계가 약함")
    st.write("- **IQR**: 데이터의 중간 50%가 들어가는 범위, 이 범위의 1.5배를 벗어나면 이상치로 봄")

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
                # 시간대별 데이터 개수 확인
                if "time" in env_df.columns:
                    env_df["hour"] = env_df["time"].dt.hour
                    hour_counts = env_df["hour"].value_counts().sort_index()
                    missing_hours = [h for h in range(24) if h not in hour_counts.index]
                    if missing_hours:
                        summary_rows.append({
                            "데이터": "환경 데이터",
                            "행 수": len(env_df),
                            "측정 항목 수": env_count,
                            "측정 기간": f"{time_min.strftime('%Y-%m-%d') if time_min is not None else 'N/A'} ~ {time_max.strftime('%Y-%m-%d') if time_max is not None else 'N/A'}",
                            "특이사항": f"일부 시간대({len(missing_hours)}개) 측정 누락"
                        })
                    else:
                        summary_rows.append({
                            "데이터": "환경 데이터",
                            "행 수": len(env_df),
                            "측정 항목 수": env_count,
                            "측정 기간": f"{time_min.strftime('%Y-%m-%d') if time_min is not None else 'N/A'} ~ {time_max.strftime('%Y-%m-%d') if time_max is not None else 'N/A'}"
                        })
                else:
                    summary_rows.append({
                        "데이터": "환경 데이터",
                        "행 수": len(env_df),
                        "측정 항목 수": env_count,
                        "측정 기간": "N/A"
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

            with col1:
                if not growth_df.empty:
                    safe_metric("총 개체수", len(growth_df))
                else:
                    st.info("📌 개체수 데이터가 없어요.")

            with col2:
                if not env_df.empty and "temperature" in env_df.columns:
                    t = env_df["temperature"].dropna()
                    if len(t) >= 2:
                        safe_metric("평균 온도 (°C)", round(t.mean(), 1))
                    else:
                        st.info("📌 온도 데이터가 부족해요.")
                else:
                    st.info("📌 온도 데이터가 없어요.")

            with col3:
                if not env_df.empty and "humidity" in env_df.columns:
                    h = env_df["humidity"].dropna()
                    if len(h) >= 2:
                        safe_metric("평균 습도 (%)", round(h.mean(), 1))
                    else:
                        st.info("📌 습도 데이터가 부족해요.")
                else:
                    st.info("📌 습도 데이터가 없어요.")

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

            # 환경 요약 표
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
                            "최대": round(vals.max(), 2),
                            "변동폭": round(vals.max() - vals.min(), 2)
                        })
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), width="stretch")

                    # 환경 데이터 특성에 따른 인사이트
                    st.subheader("🌡️ 환경 데이터가 알려주는 것")
                    temp_vals = env_df["temperature"].dropna() if "temperature" in env_df.columns else pd.Series()
                    hum_vals = env_df["humidity"].dropna() if "humidity" in env_df.columns else pd.Series()

                    if len(temp_vals) >= 2:
                        temp_range = temp_vals.max() - temp_vals.min()
                        if temp_range > 10:
                            insight_box(
                                "온도 변동이 큰 환경",
                                f"측정 기간 동안 온도가 **{round(temp_range, 1)}°C**나 변했어요. "
                                f"이런 큰 온도 변화는 식물의 생육 속도에 영향을 줄 수 있어요. "
                                f"특히 낮과 밤의 온도 차(일교차)가 큰 날이 있었는지 확인해 보세요.",
                                "note"
                            )
                        else:
                            insight_box(
                                "온도 변화가 비교적 작은 환경",
                                f"측정 기간 동안 온도 변동이 **{round(temp_range, 1)}°C**로 비교적 안정적이에요. "
                                f"식물이 일정한 환경에서 자랐다고 볼 수 있어요.",
                                "good"
                            )

                    if len(hum_vals) >= 2:
                        hum_range = hum_vals.max() - hum_vals.min()
                        avg_hum = hum_vals.mean()
                        if avg_hum < 40:
                            insight_box(
                                "건조한 환경",
                                f"평균 습도가 **{round(avg_hum, 1)}%**로 비교적 건조한 편이에요. "
                                f"극지식물은 건조한 환경에 적응한 식물이지만, 습도가 너무 낮으면 생육에 영향을 줄 수 있어요.",
                                "warn"
                            )
                        elif avg_hum > 70:
                            insight_box(
                                "습한 환경",
                                f"평균 습도가 **{round(avg_hum, 1)}%**로 높은 편이에요. "
                                f"습한 환경은 생육에 도움이 될 수 있지만, 너무 높으면 병해이 생길 수도 있어요.",
                                "info"
                            )
                        else:
                            insight_box(
                                "적당한 습도 환경",
                                f"평균 습도가 **{round(avg_hum, 1)}%**로 식물이 자라기에 대체로 적당한 범위예요.",
                                "good"
                            )

                    if "ph" in env_df.columns:
                        ph_vals = env_df["ph"].dropna()
                        if len(ph_vals) >= 2:
                            avg_ph = ph_vals.mean()
                            if avg_ph < 6.5:
                                insight_box(
                                    "약산성 환경",
                                    f"평균 pH가 **{round(avg_ph, 2)}**로 약산성이에요. "
                                    f"대부분 식물은 pH 6.0~7.0에서 잘 자라요.",
                                    "info"
                                )
                            elif avg_ph > 7.5:
                                insight_box(
                                    "약알칼리성 환경",
                                    f"평균 pH가 **{round(avg_ph, 2)}**로 약알칼리성이에요.",
                                    "info"
                                )
                            else:
                                insight_box(
                                    "중성에 가까운 환경",
                                    f"평균 pH가 **{round(avg_ph, 2)}**로 중성 부근이에요. 식물이 자라기에 좋은 토양 조건이에요.",
                                    "good"
                                )

                else:
                    st.info("📌 요약할 환경 데이터가 부족해요.")
            else:
                st.info("📌 환경 데이터에서 인식할 수 있는 열이 없어요.")

            # 날짜별 시계열
            st.subheader("날짜별 환경 변화 (일별 평균)")
            if "time" in env_df.columns:
                env_df_daily = env_df.copy()
                env_df_daily["date"] = env_df_daily["time"].dt.date
                daily = env_df_daily.groupby("date").mean(numeric_only=True)

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

                    # 온도 패턴 분석
                    temp_daily = daily["temperature"]
                    if len(temp_daily) >= 3:
                        max_day = temp_daily.idxmax()
                        min_day = temp_daily.idxmin()
                        insight_box(
                            "가장 덥고 추운 날",
                            f"가장 높았던 날: **{max_day}** ({round(temp_daily.max(), 1)}°C)<br>"
                            f"가장 낮았던 날: **{min_day}** ({round(temp_daily.min(), 1)}°C)<br>"
                            f"날짜별 변동폭: **{round(temp_daily.max() - temp_daily.min(), 1)}°C**",
                            "info"
                        )

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

                ph_ec_cols = [c for c in ["ph", "ec"] if c in daily.columns]
                if ph_ec_cols:
                    fig_ph_ec = make_subplots(
                        rows=len(ph_ec_cols), cols=1,
                        subplot_titles=["pH 변화", "EC 변화"] if len(ph_ec_cols) == 2 else [f"{ph_ec_cols[0].upper()} 변화"]
                    )
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

            # 데이터 개수 기반 안내
            n_individuals = len(growth_df)
            if n_individuals < 10:
                insight_box(
                    "데이터 개수 안내",
                    f"현재 **{n_individuals}개체**의 데이터가 있어요. 개체 수가 적어서 전체 경향을 일반화하기엔 한계가 있어요. "
                    f"더 많은 개체를 측정하면 더 믿을 수 있는 결과를 얻을 수 있어요.",
                    "warn"
                )
            elif n_individuals < 30:
                insight_box(
                    "데이터 개수 안내",
                    f"현재 **{n_individuals}개체**의 데이터가 있어요. 중간 정도의 표본 크기예요. "
                    f" 경향을 참고할 수 있지만, 개체 수가 더 많으면 더 정확한 분석이 가능해요.",
                    "info"
                )

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

                    # CV 해석
                    cv_val = (gw.std() / gw.mean() * 100) if gw.mean() > 0 else 0
                    st.subheader("📈 CV(변동계수)가 알려주는 것")
                    if cv_val < 20:
                        insight_box(
                            "개체 간 차이가 작음",
                            f"CV가 **{round(cv_val, 1)}%**로 낮은 편이에요. "
                            f"대부분의 개체가 비슷한 무게로 자랐다는 뜻이에요. "
                            f"같은 환경에서 비슷한 조건으로 자랐다고 볼 수 있어요.",
                            "good"
                        )
                    elif cv_val < 50:
                        insight_box(
                            "개체 간 차이가 중간 정도",
                            f"CV가 **{round(cv_val, 1)}%**로 중간 정도예요. "
                            f"개체에 따라 생육 차이가 나타나고 있어요. "
                            f"어떤 개체는 잘 자라고 어떤 개체는 덜 자랐는지 확인해 보세요.",
                            "note"
                        )
                    else:
                        insight_box(
                            "개체 간 차이가 매우 큼",
                            f"CV가 **{round(cv_val, 1)}%**로 높은 편이에요. "
                            f"개체 간 생육 차이가 매우 크다는 뜻이에요. "
                            f"잘 자란 개체와 그렇지 않은 개체를 비교하면 원인을 찾을 수 있을 거예요.",
                            "warn"
                        )
                else:
                    st.info("📌 생중량 데이터가 부족해요.")

            # 개체별 생육 비교 (막대 그래프)
            st.subheader("개체별 생육 비교")
            growth_cols_bar = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"] if c in growth_df.columns]
            if growth_cols_bar:
                sel_col = st.selectbox(
                    "📌 비교할 측정 항목 선택",
                    growth_cols_bar,
                    index=0 if "생중량" in growth_cols_bar else 0
                )
                vals = growth_df[sel_col].dropna()
                if len(vals) >= 2:
                    q75 = vals.quantile(0.75) if sel_col == "생중량" else None
                    q25 = vals.quantile(0.25) if sel_col == "생중량" else None

                    fig_bar = go.Figure()
                    for i, (idx, row) in enumerate(growth_df.iterrows()):
                        v = row[sel_col]
                        if pd.isna(v):
                            continue
                        if q75 is not None and v >= q75:
                            color = "#e74c3c"
                        elif q25 is not None and v <= q25:
                            color = "#3498db"
                        else:
                            color = "#2ecc71"
                        fig_bar.add_trace(go.Bar(
                            x=[row["개체번호"] if "개체번호" in growth_df.columns else i],
                            y=[v],
                            name=str(row["개체번호"] if "개체번호" in growth_df.columns else i),
                            marker_color=color,
                            showlegend=False
                        ))
                    fig_bar = fig_layout(fig_bar)
                    title_add = ""
                    if sel_col == "생중량":
                        title_add = f" (🔴 상위 25% 이상, 🔵 하위 25% 이하)"
                    fig_bar.update_layout(
                        title=f"개체별 {sel_col} 비교{title_add}",
                        xaxis_title="개체번호",
                        yaxis_title=sel_col,
                        height=400,
                        bargap=0.15
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                    # 선택한 항목에 대한 설명
                    if sel_col == "생중량":
                        insight_box(
                            "개체별 생중량 차이",
                            describe_distribution(growth_df["생중량"], "생중량", " g"),
                            "info"
                        )

                        # 상위·하위 25% 비교
                        if len(vals) >= 4:
                            top_ids = growth_df.loc[vals.sort_values(ascending=False).head(int(np.ceil(len(vals)*0.25))).index, "개체번호"] if "개체번호" in growth_df.columns else "?"
                            bottom_ids = growth_df.loc[vals.sort_values().head(int(np.ceil(len(vals)*0.25))).index, "개체번호"] if "개체번호" in growth_df.columns else "?"

                            top_avg = vals.sort_values(ascending=False).head(int(np.ceil(len(vals)*0.25))).mean()
                            bottom_avg = vals.sort_values().head(int(np.ceil(len(vals)*0.25))).mean()
                            ratio = top_avg / bottom_avg if bottom_avg > 0 else 0

                            st.subheader("🔍 상위 25% vs 하위 25% 비교")
                            c_t, c_b = st.columns(2)
                            with c_t:
                                st.write(f"**상위 25% 개체**")
                                if "개체번호" in growth_df.columns:
                                    st.write(f"고른 개체: **{list(top_ids)}**")
                                st.metric("평균 생중량", round(top_avg, 2), delta=f"전체 평균의 {round(top_avg/gw.mean()*100, 1)}%")
                            with c_b:
                                st.write(f"**하위 25% 개체**")
                                if "개체번호" in growth_df.columns:
                                    st.write(f"작은 개체: **{list(bottom_ids)}**")
                                st.metric("평균 생중량", round(bottom_avg, 2), delta=f"전체 평균의 {round(bottom_avg/gw.mean()*100, 1)}%")

                            st.info(
                                f"📌 상위 25% 개체의 평균 생중량({round(top_avg, 2)} g)은 "
                                f"하위 25% 개체({round(bottom_avg, 2)} g)의 약 **{round(ratio, 1)}배**예요. "
                                f"두 그룹 간 차이가 뚜렷하게 나타나요."
                            )
                    elif sel_col == "잎 수":
                        insight_box(
                            "잎 수 분포",
                            describe_distribution(growth_df["잎 수"], "잎 수", ""),
                            "info"
                        )
                    elif sel_col in ("지상부 길이", "지하부 길이"):
                        insight_box(
                            f"{sel_col} 분포",
                            describe_distribution(growth_df[sel_col], sel_col, " mm"),
                            "info"
                        )
                else:
                    st.info("📌 선택한 항목의 데이터가 부족해요.")
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

                    # IQR 정보
                    q1 = gw.quantile(0.25)
                    q3 = gw.quantile(0.75)
                    iqr = q3 - q1
                    st.info(
                        f"📌 박스플롯의 **상자(IQR)**는 중간 50% 개체의 범위를 나타내요: "
                        f"{round(q1, 2)} g ~ {round(q3, 2)} g (IQR = {round(iqr, 2)} g). "
                        f"상자 밖 점은 일반적인 범위에서 벗어난 이상치예요."
                    )
                else:
                    st.info("📌 생중량 데이터가 부족해서 박스플롯을 만들 수 없어요.")
            else:
                st.info("📌 생중량 열이 없어서 분포 그래프를 만들 수 없어요.")

            # 상관관계 분석
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
                        slope, intercept = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = slope * x_line + intercept

                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            name="데이터",
                            marker=dict(size=10, color="#3498db",
                                       line=dict(width=1, color="#2980b9"))
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

                        abs_r = abs(r)
                        if abs_r >= 0.7:
                            strength = "강한"
                            kind = "good" if r > 0 else "info"
                        elif abs_r >= 0.4:
                            strength = "중간 정도"
                            kind = "info"
                        elif abs_r >= 0.2:
                            strength = "약한"
                            kind = "warn"
                        else:
                            strength = "매우 약한"
                            kind = "warn"

                        direction = "양의" if r > 0 else "음의"
                        insight_box(
                            f"{x_col}과 {y_col}의 관계",
                            f"상관계수 **r = {round(r, 3)}**로 **{strength} {direction} 상관관계**예요.<br>"
                            f"- r이 양수: {x_col}이 Increase할수록 {y_col}도 증가하는 경향<br>"
                            f"- r이 음수: {x_col}이 Increase할수록 {y_col}은 감소하는 경향<br>"
                            + (f"- 두 변수 사이에 뚜렷한 관계가 관찰돼요." if abs_r >= 0.4 else
                               f"- 두 변수 사이의 관계는 약한 편이라 다른 요인도 영향을 줄 수 있어요."),
                            kind
                        )
                    else:
                        st.info(f"📌 {x_col}과 {y_col} 데이터가 부족해서 상관관계를 계산할 수 없어요.")
                else:
                    missing = [c for c in [x_col, y_col] if c not in growth_df.columns]
                    st.info(f"📌 {', '.join(missing)} 열이 없어서 이 상관분석은 건너뛰었어요.")

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
