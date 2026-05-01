import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Nordic Energy Weather Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Monthly temperature normals 1991-2020 (°C) — hardcoded for reliability
NORMALS = {
    "Helsinki":   [-3.9, -4.3, -1.0,  5.5, 11.5, 16.2, 19.0, 17.4, 12.0,  6.2,  1.2, -2.2],
    "Stockholm":  [-1.2, -1.8,  2.3,  7.8, 13.5, 17.8, 20.6, 19.3, 13.4,  7.8,  3.0, -0.5],
    "Oslo":       [-4.3, -3.9,  1.0,  6.0, 12.0, 16.5, 19.5, 18.2, 12.0,  6.4,  1.0, -3.0],
    "Copenhagen": [ 2.1,  2.0,  4.8,  9.5, 14.5, 18.0, 20.4, 19.8, 14.5,  9.7,  5.2,  2.8],
    "Oulu":       [-9.5, -9.8, -4.5,  2.8,  9.5, 15.3, 18.2, 15.8,  9.5,  2.8, -3.0, -7.5],
}

CITIES = {
    "Helsinki":   (60.17, 24.94),
    "Stockholm":  (59.33, 18.07),
    "Oslo":       (59.91, 10.75),
    "Copenhagen": (55.68, 12.57),
    "Oulu":       (65.01, 25.47),
}

# Wind sites: major Nordic offshore/onshore wind regions
WIND_SITES = {
    "DK-West":  (56.0, 8.0),
    "SE-South": (56.5, 14.0),
    "NO-South": (58.5, 7.5),
    "FI-West":  (63.0, 22.0),
}

AREA_LABELS = {"FI": "Finland", "SE3": "Sweden (SE3)", "NO2": "Norway (NO2)", "DK1": "Denmark (DK1)"}
CITY_COLORS = {
    "Helsinki": "#00b4d8", "Stockholm": "#f4d03f",
    "Oslo": "#2ecc71", "Copenhagen": "#ff6b35", "Oulu": "#e63946",
}
AREA_COLORS = {"FI": "#00b4d8", "SE3": "#f4d03f", "NO2": "#2ecc71", "DK1": "#ff6b35"}

PLOT_LAYOUT = dict(
    paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
    font_color="#e0e0e0", margin=dict(l=0, r=10, t=30, b=0),
    legend=dict(bgcolor="#1a1f2e", bordercolor="#2a2f3e", borderwidth=1),
    xaxis=dict(gridcolor="#2a2f3e", showgrid=True),
    yaxis=dict(gridcolor="#2a2f3e", showgrid=True),
)


# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nao() -> pd.DataFrame:
    url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
           "norm.nao.monthly.b5001.current.ascii.table")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        rows = []
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) < 13:
                continue
            year = int(parts[0])
            for m, v in enumerate(parts[1:13], 1):
                val = float(v)
                if val != -99.9:
                    rows.append({"date": pd.Timestamp(year=year, month=m, day=1), "nao": val})
        df = pd.DataFrame(rows)
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=36)
        return df[df["date"] >= cutoff].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "nao"])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_temperature(days: int = 365) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    rows = []
    for city, (lat, lon) in CITIES.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
            f"&daily=temperature_2m_mean&timezone=auto"
        )
        try:
            data = requests.get(url, timeout=20).json()
            for d, t in zip(data["daily"]["time"], data["daily"]["temperature_2m_mean"]):
                if t is not None:
                    rows.append({"date": pd.Timestamp(d), "city": city, "temp": float(t)})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wind(days: int = 90) -> pd.DataFrame:
    """Wind speed at 100 m → approximate capacity factor via simplified power curve."""
    end = datetime.now()
    start = end - timedelta(days=days)
    rows = []
    for site, (lat, lon) in WIND_SITES.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
            f"&daily=windspeed_10m_max&timezone=auto"
        )
        try:
            data = requests.get(url, timeout=20).json()
            for d, ws in zip(data["daily"]["time"], data["daily"]["windspeed_10m_max"]):
                if ws is not None:
                    ws100 = float(ws) * 1.35          # rough 10→100 m correction
                    cf = wind_cf(ws100)
                    rows.append({"date": pd.Timestamp(d), "site": site, "wind_speed": ws100, "cf": cf})
        except Exception:
            pass
    return pd.DataFrame(rows)


def wind_cf(ws: float) -> float:
    """Simplified wind turbine power curve → capacity factor 0–1."""
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    if ws < cut_in or ws >= cut_out:
        return 0.0
    if ws >= rated:
        return 1.0
    return ((ws - cut_in) / (rated - cut_in)) ** 3


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_prices(days: int = 90) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    url = (
        "https://api.energidataservice.dk/dataset/Elspotprices"
        f"?limit=100000"
        f'&filter={{"PriceArea":["DK1","FI","NO2","SE3"]}}'
        f"&start={start:%Y-%m-%dT00:00}&end={end:%Y-%m-%dT00:00}"
        "&sort=HourDK%20ASC"
    )
    try:
        data = requests.get(url, timeout=20).json()
        df = pd.DataFrame(data.get("records", []))
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["HourDK"])
        df["price"] = pd.to_numeric(df["SpotPriceEUR"], errors="coerce")
        return df[["date", "PriceArea", "price"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["date", "PriceArea", "price"])


# ── Derived metrics ────────────────────────────────────────────────────────────

def add_hdd_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Add HDD and deviation-from-normal columns."""
    if df.empty:
        return df
    df = df.copy()
    df["month"] = df["date"].dt.month
    normal = df.apply(lambda r: NORMALS[r["city"]][r["month"] - 1], axis=1)
    df["hdd"] = (17 - df["temp"]).clip(lower=0)
    df["hdd_normal"] = (17 - normal).clip(lower=0)
    df["temp_dev"] = df["temp"] - normal
    df["hdd_dev"] = df["hdd"] - df["hdd_normal"]
    df = df.sort_values(["city", "date"])
    df["hdd_dev_30d"] = df.groupby("city")["hdd_dev"].transform(
        lambda s: s.rolling(30, min_periods=15).mean()
    )
    df["temp_dev_7d"] = df.groupby("city")["temp_dev"].transform(
        lambda s: s.rolling(7, min_periods=3).mean()
    )
    return df


def risk_score(nao: float | None, hdd_dev: float | None, price: float | None) -> float:
    score = 5.0
    if nao is not None:
        score -= np.clip(nao, -3, 3)
    if hdd_dev is not None:
        score += np.clip(hdd_dev * 0.35, -2, 2.5)
    if price is not None:
        if price > 150:
            score += 1.5
        elif price > 100:
            score += 0.5
        elif price < 25:
            score -= 1.0
    return float(np.clip(score, 0, 10))


def risk_color(s: float) -> str:
    if s >= 7.5:
        return "#e63946"
    if s >= 5.5:
        return "#ff6b35"
    if s >= 3.5:
        return "#f4d03f"
    return "#2ecc71"


def risk_label(s: float) -> str:
    if s >= 7.5:
        return "CRITICAL"
    if s >= 5.5:
        return "HIGH"
    if s >= 3.5:
        return "MODERATE"
    return "LOW"


# ── Helpers ────────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sublabel: str, color: str, note: str = "") -> str:
    return f"""
    <div style="background:#1a1f2e;border-radius:12px;padding:18px 20px;
                border-left:4px solid {color};height:100%">
        <div style="color:#8b8fa8;font-size:11px;font-weight:700;
                    letter-spacing:.8px;text-transform:uppercase">{label}</div>
        <div style="color:{color};font-size:30px;font-weight:700;
                    line-height:1.2;margin:6px 0">{value}</div>
        <div style="color:{color};font-size:12px">{sublabel}</div>
        <div style="color:#8b8fa8;font-size:11px;margin-top:4px">{note}</div>
    </div>"""


def section(title: str):
    st.markdown(
        f'<div style="color:#e0e0e0;font-size:17px;font-weight:600;'
        f'margin:28px 0 12px;border-bottom:1px solid #2a2f3e;padding-bottom:8px">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )


# ── App ────────────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="margin-bottom:4px">⚡ Nordic Energy Weather Monitor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="color:#8b8fa8;font-size:12px;margin-bottom:20px">'
    f'Updated {datetime.now():%d.%m.%Y %H:%M} · Data: NOAA · Open-Meteo · Energi Data Service'
    f'</div>',
    unsafe_allow_html=True,
)

# Fetch
with st.spinner("Loading data..."):
    nao_df   = fetch_nao()
    temp_raw = fetch_temperature(365)
    wind_df  = fetch_wind(90)
    price_df = fetch_prices(90)

temp_df = add_hdd_deviation(temp_raw)

# Current values
latest_nao  = nao_df["nao"].iloc[-1]  if not nao_df.empty  else None
prev_nao    = nao_df["nao"].iloc[-2]  if len(nao_df) >= 2  else None
nao_month   = nao_df["date"].iloc[-1].strftime("%m/%Y") if not nao_df.empty else "–"

recent_temp = temp_df[temp_df["date"] >= datetime.now() - timedelta(days=30)] if not temp_df.empty else pd.DataFrame()
hdd_dev_avg = recent_temp["hdd_dev"].mean() if not recent_temp.empty else None

recent_wind = wind_df[wind_df["date"] >= datetime.now() - timedelta(days=30)] if not wind_df.empty else pd.DataFrame()
cf_avg      = recent_wind["cf"].mean() if not recent_wind.empty else None

recent_price = price_df[price_df["date"] >= datetime.now() - timedelta(days=7)] if not price_df.empty else pd.DataFrame()
price_avg    = recent_price["price"].mean() if not recent_price.empty else None
prev_week    = price_df[
    (price_df["date"] >= datetime.now() - timedelta(days=14)) &
    (price_df["date"] <  datetime.now() - timedelta(days=7))
]["price"].mean() if not price_df.empty else None

rs = risk_score(latest_nao, hdd_dev_avg, price_avg)

# ── KPI Row ────────────────────────────────────────────────────────────────────

section("Current Status")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    val = f"{latest_nao:+.2f}" if latest_nao is not None else "–"
    delta = f"{latest_nao - prev_nao:+.2f} prev. month" if latest_nao is not None and prev_nao is not None else ""
    col = "#e63946" if latest_nao is not None and latest_nao < -1 else \
          "#ff6b35" if latest_nao is not None and latest_nao < 0 else \
          "#2ecc71" if latest_nao is not None and latest_nao > 0.5 else "#00b4d8"
    lbl = "Negative" if latest_nao is not None and latest_nao < -0.5 else \
          "Positive" if latest_nao is not None and latest_nao > 0.5 else "Neutral"
    st.markdown(kpi_card(f"NAO ({nao_month})", val, lbl, col, delta), unsafe_allow_html=True)

with c2:
    val  = f"{hdd_dev_avg:+.1f} HDD/day" if hdd_dev_avg is not None else "–"
    col  = "#e63946" if hdd_dev_avg is not None and hdd_dev_avg > 2 else \
           "#2ecc71" if hdd_dev_avg is not None and hdd_dev_avg < -2 else "#00b4d8"
    lbl  = "Colder than normal" if hdd_dev_avg is not None and hdd_dev_avg > 1 else \
           "Warmer than normal" if hdd_dev_avg is not None and hdd_dev_avg < -1 else "Near normal"
    st.markdown(kpi_card("HDD Deviation 30d", val, lbl, col, "avg. 5 cities"), unsafe_allow_html=True)

with c3:
    val = f"{cf_avg*100:.0f} %" if cf_avg is not None else "–"
    col = "#e63946" if cf_avg is not None and cf_avg < 0.15 else \
          "#2ecc71" if cf_avg is not None and cf_avg > 0.35 else "#00b4d8"
    lbl = "Low wind" if cf_avg is not None and cf_avg < 0.15 else \
          "Good wind" if cf_avg is not None and cf_avg > 0.35 else "Moderate"
    st.markdown(kpi_card("Wind Capacity 30d", val, lbl, col, "avg. 4 regions"), unsafe_allow_html=True)

with c4:
    val   = f"{price_avg:.0f} €/MWh" if price_avg is not None else "–"
    delta = f"{price_avg - prev_week:+.0f} prev. week" if price_avg is not None and prev_week is not None else ""
    col   = "#e63946" if price_avg is not None and price_avg > 150 else \
            "#ff6b35" if price_avg is not None and price_avg > 80  else \
            "#2ecc71" if price_avg is not None and price_avg < 30  else "#00b4d8"
    lbl   = "High" if price_avg is not None and price_avg > 150 else \
            "Moderate" if price_avg is not None and price_avg > 50 else "Low"
    st.markdown(kpi_card("Spot Price 7d avg", val, lbl, col, delta), unsafe_allow_html=True)

with c5:
    col = risk_color(rs)
    lbl = risk_label(rs)
    st.markdown(kpi_card("Overall Risk", f"{rs:.1f}/10", lbl, col, "NAO + HDD + price"), unsafe_allow_html=True)

# ── NAO ───────────────────────────────────────────────────────────────────────

section("NAO Index — 36 months")

if not nao_df.empty:
    fig = go.Figure()
    fig.add_hrect(y0=-5, y1=-1, fillcolor="rgba(230,57,70,.12)", line_width=0,
                  annotation_text="High risk (NAO < −1)", annotation_position="top left",
                  annotation_font_color="#e63946", annotation_font_size=11)
    fig.add_hrect(y0=1, y1=5, fillcolor="rgba(46,204,113,.07)", line_width=0,
                  annotation_text="Low risk (NAO > +1)", annotation_position="bottom right",
                  annotation_font_color="#2ecc71", annotation_font_size=11)
    fig.add_hline(y=0,    line_dash="dot",  line_color="#8b8fa8", line_width=1)
    fig.add_hline(y=-1.0, line_dash="dash", line_color="#e63946", line_width=1)
    bar_colors = ["#e63946" if v < 0 else "#2ecc71" for v in nao_df["nao"]]
    fig.add_trace(go.Bar(
        x=nao_df["date"], y=nao_df["nao"], marker_color=bar_colors,
        hovertemplate="%{x|%m/%Y}: %{y:+.2f}<extra></extra>",
    ))
    fig.update_layout({**PLOT_LAYOUT, "height": 320, "showlegend": False,
                       "yaxis": dict(range=[-4, 4], gridcolor="#2a2f3e", showgrid=True)})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("NAO data unavailable.")

# ── Temperature deviation ─────────────────────────────────────────────────────

section("Temperature Deviation from Normal — 7-day rolling avg")

if not temp_df.empty:
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dot", line_color="#8b8fa8", line_width=1)
    for city in CITIES:
        cd = temp_df[temp_df["city"] == city].sort_values("date")
        fig.add_trace(go.Scatter(
            x=cd["date"], y=cd["temp_dev_7d"], name=city,
            line=dict(color=CITY_COLORS[city], width=2), mode="lines",
            hovertemplate=f"{city}: %{{y:+.1f}}°C<extra></extra>",
        ))
    fig.update_layout({**PLOT_LAYOUT, "height": 320,
                       "yaxis": dict(gridcolor="#2a2f3e", ticksuffix="°C",
                                     title="Deviation from normal")})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Temperature data unavailable.")

# ── Wind CF ───────────────────────────────────────────────────────────────────

section("Wind Capacity Factor — 7-day rolling avg")

if not wind_df.empty:
    site_colors = {
        "DK-West": "#ff6b35", "SE-South": "#f4d03f",
        "NO-South": "#2ecc71", "FI-West":  "#00b4d8",
    }
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.15, fillcolor="rgba(230,57,70,.08)", line_width=0,
                  annotation_text="Low wind output", annotation_position="top right",
                  annotation_font_color="#e63946", annotation_font_size=11)
    for site in WIND_SITES:
        sd = wind_df[wind_df["site"] == site].sort_values("date")
        sd = sd.copy()
        sd["cf_7d"] = sd["cf"].rolling(7, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=sd["date"], y=sd["cf_7d"] * 100, name=site,
            line=dict(color=site_colors.get(site, "#fff"), width=2), mode="lines",
            hovertemplate=f"{site}: %{{y:.1f}} %<extra></extra>",
        ))
    fig.update_layout({**PLOT_LAYOUT, "height": 300,
                       "yaxis": dict(gridcolor="#2a2f3e", ticksuffix=" %",
                                     title="Capacity factor", range=[0, 100])})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Wind data unavailable.")

# ── Spot prices ───────────────────────────────────────────────────────────────

section("Electricity Spot Prices — 90 days")

if not price_df.empty:
    daily = (
        price_df.groupby([price_df["date"].dt.date, "PriceArea"])["price"]
        .mean().reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    fig = go.Figure()
    p_max = daily["price"].max()
    if p_max > 80:
        fig.add_hrect(y0=150, y1=max(200, p_max * 1.1),
                      fillcolor="rgba(230,57,70,.08)", line_width=0,
                      annotation_text="High price (>150 €)",
                      annotation_position="top right",
                      annotation_font_color="#e63946", annotation_font_size=11)
    for area in ["FI", "SE3", "NO2", "DK1"]:
        ad = daily[daily["PriceArea"] == area].sort_values("date")
        if ad.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ad["date"], y=ad["price"],
            name=AREA_LABELS.get(area, area),
            line=dict(color=AREA_COLORS[area], width=2), mode="lines",
            hovertemplate=f"{AREA_LABELS.get(area, area)}: %{{y:.0f}} €/MWh<extra></extra>",
        ))
    fig.update_layout({**PLOT_LAYOUT, "height": 320,
                       "yaxis": dict(gridcolor="#2a2f3e", ticksuffix=" €",
                                     title="EUR/MWh")})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Price data unavailable.")

# ── Scenario matching ─────────────────────────────────────────────────────────

section("Scenario Comparison — which situation does the current data resemble?")

SCENARIOS = [
    dict(name="Stress: cold + low wind", risk=9.2, color="#e63946",
         desc="Negative NAO, frost, wind drought, low reservoirs",
         nao="< −1.0", hdd="> +3 HDD/day", wind="< 15 %", price="> 150 €/MWh"),
    dict(name="Base scenario", risk=4.0, color="#00b4d8",
         desc="SSP2-4.5 central path, normal conditions",
         nao="−0.5 – +0.5", hdd="±1 HDD/day", wind="25–35 %", price="50–90 €/MWh"),
    dict(name="Wet & windy", risk=1.8, color="#2ecc71",
         desc="Positive NAO, mild, windy, high reservoirs",
         nao="> +1.0", hdd="< −2 HDD/day", wind="> 40 %", price="< 40 €/MWh"),
    dict(name="Structural warming", risk=6.5, color="#f4d03f",
         desc="SSP5-8.5 high-emissions path, growing uncertainty",
         nao="variable", hdd="declining trend", wind="uncertain", price="structural shift"),
]

# Which scenario best matches current risk?
diffs = [abs(rs - s["risk"]) for s in SCENARIOS]
best_match = diffs.index(min(diffs))

cols = st.columns(4)
for i, s in enumerate(SCENARIOS):
    badge = " ← CURRENT" if i == best_match else ""
    with cols[i]:
        st.markdown(f"""
        <div style="background:#1a1f2e;border-radius:12px;padding:16px;
                    border-left:4px solid {s['color']};">
            <div style="color:#8b8fa8;font-size:11px;font-weight:700;
                        text-transform:uppercase;letter-spacing:.6px">{s['name']}</div>
            <div style="color:{s['color']};font-size:22px;font-weight:700;
                        margin:6px 0">Risk {s['risk']}/10{badge}</div>
            <div style="color:#8b8fa8;font-size:11px;margin-bottom:10px">{s['desc']}</div>
            <div style="color:#e0e0e0;font-size:12px">NAO: {s['nao']}</div>
            <div style="color:#e0e0e0;font-size:12px">HDD: {s['hdd']}</div>
            <div style="color:#e0e0e0;font-size:12px">Wind: {s['wind']}</div>
            <div style="color:#e0e0e0;font-size:12px">Price: {s['price']}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="color:#8b8fa8;font-size:11px">'
    'Data: NOAA CPC (NAO index) · Open-Meteo Archive API (temperature, wind) · '
    'Energi Data Service / Energinet (spot prices, areas FI/SE3/NO2/DK1) · '
    'Wind capacity factor estimated via simplified power curve · '
    'Analysis framework: Nordic Weather Futures'
    '</div>',
    unsafe_allow_html=True,
)
