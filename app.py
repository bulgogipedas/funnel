import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats

# ==================================
# CONFIG
# ==================================
DATA_DIR = Path("data")

st.set_page_config(
    page_title="Sales Funnel Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
    
    .main { background-color: #0e1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #151922 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f2f6;
        margin: 0;
    }
    
    .metric-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-delta-positive { color: #3fb950; font-size: 0.9rem; }
    .metric-delta-negative { color: #f85149; font-size: 0.9rem; }
    
    .insight-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
    }
    
    .insight-box-warning {
        background: linear-gradient(135deg, #5c3d1e 0%, #3d2a14 100%);
        border-left: 4px solid #d29922;
    }
    
    .insight-box-success {
        background: linear-gradient(135deg, #1e4620 0%, #142d16 100%);
        border-left: 4px solid #3fb950;
    }
    
    .insight-title {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: #f0f2f6;
        margin-bottom: 8px;
    }
    
    .insight-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: #c9d1d9;
        line-height: 1.5;
    }
    
    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f2f6;
        border-bottom: 2px solid #30363d;
        padding-bottom: 12px;
        margin: 32px 0 24px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #8b949e;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


# ==================================
# DATA LOADING
# ==================================
@st.cache_data
def load_tables(base_path: Path = DATA_DIR):
    user = pd.read_csv(base_path / "user_table.csv", parse_dates=["date"])
    home = pd.read_csv(base_path / "home_page_table.csv")
    search = pd.read_csv(base_path / "search_page_table.csv")
    payment = pd.read_csv(base_path / "payment_page_table.csv")
    confirm = pd.read_csv(base_path / "payment_confirmation_table.csv")

    return {
        "user": user,
        "home": home,
        "search": search,
        "payment": payment,
        "confirm": confirm,
    }


def build_user_funnel_df(tables: dict) -> pd.DataFrame:
    user = tables["user"].copy()

    home_ids = set(tables["home"]["user_id"])
    search_ids = set(tables["search"]["user_id"])
    payment_ids = set(tables["payment"]["user_id"])
    confirm_ids = set(tables["confirm"]["user_id"])

    user["stage_home"] = user["user_id"].isin(home_ids)
    user["stage_search"] = user["user_id"].isin(search_ids)
    user["stage_payment"] = user["user_id"].isin(payment_ids)
    user["stage_confirm"] = user["user_id"].isin(confirm_ids)

    def stage_index(row):
        if row["stage_confirm"]:
            return 4
        if row["stage_payment"]:
            return 3
        if row["stage_search"]:
            return 2
        if row["stage_home"]:
            return 1
        return 0

    user["stage_index"] = user.apply(stage_index, axis=1)
    user["converted"] = user["stage_index"] == 4
    user["week"] = user["date"].dt.isocalendar().week
    user["month"] = user["date"].dt.to_period("M").astype(str)
    user["day_of_week"] = user["date"].dt.day_name()

    return user


# ==================================
# ANALYTICS FUNCTIONS
# ==================================
def compute_funnel_metrics(df: pd.DataFrame) -> dict:
    """Compute comprehensive funnel metrics."""
    total = len(df)
    home = (df["stage_index"] >= 1).sum()
    search = (df["stage_index"] >= 2).sum()
    payment = (df["stage_index"] >= 3).sum()
    confirm = (df["stage_index"] >= 4).sum()

    return {
        "total_users": total,
        "home_users": home,
        "search_users": search,
        "payment_users": payment,
        "confirm_users": confirm,
        "home_to_search_rate": search / home * 100 if home > 0 else 0,
        "search_to_payment_rate": payment / search * 100 if search > 0 else 0,
        "payment_to_confirm_rate": confirm / payment * 100 if payment > 0 else 0,
        "overall_conversion": confirm / home * 100 if home > 0 else 0,
        "home_dropoff": (home - search) / home * 100 if home > 0 else 0,
        "search_dropoff": (search - payment) / search * 100 if search > 0 else 0,
        "payment_dropoff": (payment - confirm) / payment * 100 if payment > 0 else 0,
    }


def compute_segment_analysis(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Deep segment analysis with statistical testing."""
    rows = []
    overall_conv = df["converted"].mean()

    for value, grp in df.groupby(segment_col):
        n = len(grp)
        home = (grp["stage_index"] >= 1).sum()
        search = (grp["stage_index"] >= 2).sum()
        payment = (grp["stage_index"] >= 3).sum()
        confirm = (grp["stage_index"] >= 4).sum()

        conv_rate = confirm / home if home > 0 else 0
        
        # Chi-square test vs rest of population
        rest = df[df[segment_col] != value]
        rest_conv = rest["converted"].sum()
        rest_total = (rest["stage_index"] >= 1).sum()

        if home > 0 and rest_total > 0:
            contingency = [[confirm, home - confirm], [rest_conv, rest_total - rest_conv]]
            try:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            except:
                p_value = 1.0
        else:
            p_value = 1.0

        rows.append({
            segment_col: value,
            "users": n,
            "funnel_entered": home,
            "reached_search": search,
            "reached_payment": payment,
            "converted": confirm,
            "conversion_rate": conv_rate * 100,
            "vs_average": (conv_rate - overall_conv) / overall_conv * 100 if overall_conv > 0 else 0,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "home_to_search": search / home * 100 if home > 0 else 0,
            "search_to_payment": payment / search * 100 if search > 0 else 0,
            "payment_to_confirm": confirm / payment * 100 if payment > 0 else 0,
        })

    return pd.DataFrame(rows).sort_values("conversion_rate", ascending=False)


def compute_cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly cohort analysis."""
    df = df.copy()
    df["cohort_week"] = df["date"].dt.strftime("%Y-W%U")

    cohorts = []
    for cohort, grp in df.groupby("cohort_week"):
        metrics = compute_funnel_metrics(grp)
        cohorts.append({
            "cohort": cohort,
            "users": metrics["home_users"],
            "conversion_rate": metrics["overall_conversion"],
            "home_to_search": metrics["home_to_search_rate"],
            "search_to_payment": metrics["search_to_payment_rate"],
            "payment_to_confirm": metrics["payment_to_confirm_rate"],
        })

    return pd.DataFrame(cohorts).sort_values("cohort")


def compute_drop_off_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze where users drop off by segment."""
    results = []

    for device in df["device"].unique():
        for sex in df["sex"].unique():
            subset = df[(df["device"] == device) & (df["sex"] == sex)]
            if len(subset) == 0:
                continue

            metrics = compute_funnel_metrics(subset)
            
            # Find the biggest drop-off point
            dropoffs = {
                "Home → Search": metrics["home_dropoff"],
                "Search → Payment": metrics["search_dropoff"],
                "Payment → Confirm": metrics["payment_dropoff"],
            }
            worst_stage = max(dropoffs, key=dropoffs.get)

            results.append({
                "device": device,
                "sex": sex,
                "users": metrics["home_users"],
                "conversion_rate": metrics["overall_conversion"],
                "biggest_dropoff_stage": worst_stage,
                "biggest_dropoff_rate": dropoffs[worst_stage],
                "home_dropoff": metrics["home_dropoff"],
                "search_dropoff": metrics["search_dropoff"],
                "payment_dropoff": metrics["payment_dropoff"],
            })

    return pd.DataFrame(results).sort_values("conversion_rate", ascending=False)


def compute_day_of_week_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze conversion by day of week."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    results = []
    for day in day_order:
        subset = df[df["day_of_week"] == day]
        if len(subset) == 0:
            continue
        metrics = compute_funnel_metrics(subset)
        results.append({
            "day": day,
            "users": metrics["home_users"],
            "conversion_rate": metrics["overall_conversion"],
            "home_to_search": metrics["home_to_search_rate"],
        })

    return pd.DataFrame(results)


def identify_key_insights(df: pd.DataFrame, segment_analysis: dict) -> list:
    """Generate actionable insights from the data."""
    insights = []
    metrics = compute_funnel_metrics(df)

    # Insight 1: Biggest drop-off point
    dropoffs = {
        "Home to Search": (metrics["home_dropoff"], metrics["home_to_search_rate"]),
        "Search to Payment": (metrics["search_dropoff"], metrics["search_to_payment_rate"]),
        "Payment to Confirmation": (metrics["payment_dropoff"], metrics["payment_to_confirm_rate"]),
    }
    worst_stage = max(dropoffs, key=lambda x: dropoffs[x][0])
    insights.append({
        "type": "warning",
        "title": f"🚨 Critical Drop-off: {worst_stage}",
        "text": f"You're losing {dropoffs[worst_stage][0]:.1f}% of users at the {worst_stage} stage. "
                f"Only {dropoffs[worst_stage][1]:.1f}% proceed to the next step. "
                f"This represents {int(metrics['home_users'] * dropoffs[worst_stage][0] / 100):,} lost users.",
        "priority": 1,
    })

    # Insight 2: Device performance gap
    device_df = segment_analysis["device"]
    if len(device_df) >= 2:
        best = device_df.iloc[0]
        worst = device_df.iloc[-1]
        gap = best["conversion_rate"] - worst["conversion_rate"]
        if gap > 0.1:  # Meaningful gap
            insights.append({
                "type": "insight",
                "title": f"📱 Device Gap: {best['device']} outperforms {worst['device']}",
                "text": f"{best['device']} users convert at {best['conversion_rate']:.2f}% vs "
                        f"{worst['device']} at {worst['conversion_rate']:.2f}%. "
                        f"{'This difference is statistically significant.' if best['significant'] else 'Consider A/B testing to validate.'} "
                        f"Optimizing {worst['device']} experience could capture {int(worst['funnel_entered'] * gap / 100):,} more conversions.",
                "priority": 2,
            })

    # Insight 3: Gender analysis
    sex_df = segment_analysis["sex"]
    if len(sex_df) >= 2:
        best = sex_df.iloc[0]
        worst = sex_df.iloc[-1]
        gap = best["conversion_rate"] - worst["conversion_rate"]
        if gap > 0.05 and best["significant"]:
            insights.append({
                "type": "insight",
                "title": f"👤 Demographic Insight: {best['sex']} converts better",
                "text": f"{best['sex']} users have {best['conversion_rate']:.2f}% conversion vs "
                        f"{worst['sex']} at {worst['conversion_rate']:.2f}% (p={best['p_value']:.4f}). "
                        f"Consider tailoring messaging or UX for {worst['sex']} audience.",
                "priority": 3,
            })

    # Insight 4: Payment abandonment
    if metrics["payment_dropoff"] > 90:
        insights.append({
            "type": "warning",
            "title": "💳 Payment Abandonment Crisis",
            "text": f"{metrics['payment_dropoff']:.1f}% of users who reach payment don't complete. "
                    f"That's {metrics['payment_users'] - metrics['confirm_users']:,} abandoned carts. "
                    f"Consider: simplifying checkout, adding trust signals, offering guest checkout, or A/B testing payment options.",
            "priority": 1,
        })

    # Insight 5: Revenue opportunity
    avg_order_value = 50  # Assumption - could be parameterized
    lost_at_payment = metrics["payment_users"] - metrics["confirm_users"]
    potential_revenue = lost_at_payment * avg_order_value
    if lost_at_payment > 100:
        insights.append({
            "type": "success",
            "title": "💰 Revenue Opportunity",
            "text": f"If you recover just 10% of payment abandonments ({int(lost_at_payment * 0.1):,} users), "
                    f"you could generate ${int(potential_revenue * 0.1):,} additional revenue (assuming ${avg_order_value} AOV). "
                    f"Focus on cart recovery emails, exit-intent popups, and checkout optimization.",
            "priority": 2,
        })

    return sorted(insights, key=lambda x: x["priority"])


# ==================================
# VISUALIZATION FUNCTIONS
# ==================================
def create_funnel_chart(metrics: dict) -> go.Figure:
    """Create an enhanced funnel visualization."""
    stages = ["Home Page", "Search Page", "Payment Page", "Confirmed"]
    values = [
        metrics["home_users"],
        metrics["search_users"],
        metrics["payment_users"],
        metrics["confirm_users"],
    ]
    
    # Calculate percentages
    percentages = [v / values[0] * 100 for v in values]
    
    fig = go.Figure()
    
    # Custom funnel with gradient colors
    colors = ["#6366f1", "#8b5cf6", "#a855f7", "#22c55e"]
    
    fig.add_trace(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        texttemplate="<b>%{value:,}</b><br>%{percentInitial:.1%}",
        marker=dict(
            color=colors,
            line=dict(width=2, color="#1a1f2e"),
        ),
        connector=dict(line=dict(color="#30363d", width=2)),
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#f0f2f6"),
        title=dict(
            text="<b>Conversion Funnel</b>",
            font=dict(size=18),
            x=0.5,
        ),
    )

    return fig


def create_dropoff_waterfall(metrics: dict) -> go.Figure:
    """Create waterfall chart showing where users drop off."""
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Entered Funnel", "Lost: Home→Search", "Lost: Search→Payment", "Lost: Payment→Confirm", "Converted"],
        y=[
            metrics["home_users"],
            -(metrics["home_users"] - metrics["search_users"]),
            -(metrics["search_users"] - metrics["payment_users"]),
            -(metrics["payment_users"] - metrics["confirm_users"]),
            0,
        ],
        text=[
            f"{metrics['home_users']:,}",
            f"-{metrics['home_users'] - metrics['search_users']:,}",
            f"-{metrics['search_users'] - metrics['payment_users']:,}",
            f"-{metrics['payment_users'] - metrics['confirm_users']:,}",
            f"{metrics['confirm_users']:,}",
        ],
        textposition="outside",
        connector=dict(line=dict(color="#30363d")),
        decreasing=dict(marker=dict(color="#f85149")),
        increasing=dict(marker=dict(color="#3fb950")),
        totals=dict(marker=dict(color="#58a6ff")),
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#f0f2f6"),
        title=dict(
            text="<b>User Drop-off Analysis</b><br><sup>Where are you losing customers?</sup>",
            font=dict(size=18),
            x=0.5,
        ),
        yaxis=dict(
            gridcolor="#21262d",
            zerolinecolor="#30363d",
        ),
        xaxis=dict(tickangle=-30),
        showlegend=False,
    )

    return fig


def create_segment_comparison(segment_df: pd.DataFrame, segment_col: str) -> go.Figure:
    """Create segment comparison chart with conversion rates and significance."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Conversion Rate by Segment", "Stage-by-Stage Conversion"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.12,
    )

    # Left: Overall conversion with significance markers
    colors = ["#3fb950" if sig else "#6366f1" for sig in segment_df["significant"]]
    
    fig.add_trace(
        go.Bar(
            x=segment_df[segment_col],
            y=segment_df["conversion_rate"],
            marker_color=colors,
            text=[f"{v:.2f}%{'*' if sig else ''}" for v, sig in zip(segment_df["conversion_rate"], segment_df["significant"])],
            textposition="outside",
            name="Conversion Rate",
        ),
        row=1, col=1,
    )

    # Right: Stage-by-stage comparison
    for i, (_, row) in enumerate(segment_df.iterrows()):
        fig.add_trace(
            go.Bar(
                x=["Home→Search", "Search→Payment", "Payment→Confirm"],
                y=[row["home_to_search"], row["search_to_payment"], row["payment_to_confirm"]],
                name=row[segment_col],
                text=[f"{v:.1f}%" for v in [row["home_to_search"], row["search_to_payment"], row["payment_to_confirm"]]],
                textposition="outside",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#f0f2f6"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
    )

    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_annotations(font_size=14)

    return fig


def create_cohort_heatmap(cohort_df: pd.DataFrame) -> go.Figure:
    """Create cohort analysis heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=[cohort_df["conversion_rate"].values],
        x=cohort_df["cohort"].values,
        y=["Conversion %"],
        colorscale=[
            [0, "#1a1f2e"],
            [0.5, "#6366f1"],
            [1, "#22c55e"],
        ],
        text=[[f"{v:.2f}%" for v in cohort_df["conversion_rate"]]],
        texttemplate="%{text}",
        textfont=dict(size=10, color="#f0f2f6"),
        hovertemplate="Week: %{x}<br>Conversion: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#f0f2f6"),
        title=dict(
            text="<b>Weekly Cohort Performance</b>",
            font=dict(size=14),
        ),
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_trend_chart(cohort_df: pd.DataFrame) -> go.Figure:
    """Create trend analysis with moving average."""
    cohort_df = cohort_df.copy()
    cohort_df["ma_3"] = cohort_df["conversion_rate"].rolling(3, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cohort_df["cohort"],
        y=cohort_df["conversion_rate"],
        mode="lines+markers",
        name="Weekly Conversion",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=8),
    ))

    fig.add_trace(go.Scatter(
        x=cohort_df["cohort"],
        y=cohort_df["ma_3"],
        mode="lines",
        name="3-Week Moving Avg",
        line=dict(color="#f59e0b", width=3, dash="dash"),
    ))

    # Add trend line
    x_numeric = np.arange(len(cohort_df))
    z = np.polyfit(x_numeric, cohort_df["conversion_rate"], 1)
    p = np.poly1d(z)
    trend_direction = "📈 Improving" if z[0] > 0 else "📉 Declining"

    fig.add_trace(go.Scatter(
        x=cohort_df["cohort"],
        y=p(x_numeric),
        mode="lines",
        name=f"Trend ({trend_direction})",
        line=dict(color="#f85149" if z[0] < 0 else "#3fb950", width=2, dash="dot"),
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#f0f2f6"),
        title=dict(
            text=f"<b>Conversion Trend Over Time</b><br><sup>{trend_direction} trend detected</sup>",
            font=dict(size=16),
            x=0.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        xaxis=dict(tickangle=-45, gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Conversion Rate (%)"),
    )

    return fig


def render_insight_box(insight: dict):
    """Render a styled insight box."""
    box_class = {
        "warning": "insight-box insight-box-warning",
        "success": "insight-box insight-box-success",
        "insight": "insight-box",
    }.get(insight["type"], "insight-box")

    st.markdown(f"""
    <div class="{box_class}">
        <div class="insight-title">{insight['title']}</div>
        <div class="insight-text">{insight['text']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = None, delta_positive: bool = True):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        delta_class = "metric-delta-positive" if delta_positive else "metric-delta-negative"
        delta_html = f'<span class="{delta_class}">{delta}</span>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ==================================
# MAIN APP
# ==================================
tables = load_tables()
user_funnel = build_user_funnel_df(tables)
metrics = compute_funnel_metrics(user_funnel)

# Segment analyses
segment_analysis = {
    "device": compute_segment_analysis(user_funnel, "device"),
    "sex": compute_segment_analysis(user_funnel, "sex"),
}
cohort_df = compute_cohort_analysis(user_funnel)
dropoff_df = compute_drop_off_analysis(user_funnel)
day_analysis = compute_day_of_week_analysis(user_funnel)
insights = identify_key_insights(user_funnel, segment_analysis)

# ==================================
# HEADER
# ==================================
st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="font-family: 'DM Sans', sans-serif; font-size: 2.5rem; font-weight: 700; color: #f0f2f6; margin: 0;">
        📊 Sales Funnel Intelligence
    </h1>
    <p style="font-family: 'DM Sans', sans-serif; font-size: 1.1rem; color: #8b949e; margin-top: 8px;">
        Actionable insights to optimize your conversion pipeline
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================
# EXECUTIVE SUMMARY METRICS
# ==================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    render_metric_card("Funnel Entries", f"{metrics['home_users']:,}")
with col2:
    render_metric_card("Conversions", f"{metrics['confirm_users']:,}")
with col3:
    render_metric_card("Conversion Rate", f"{metrics['overall_conversion']:.2f}%")
with col4:
    render_metric_card("Biggest Drop-off", f"{max(metrics['home_dropoff'], metrics['search_dropoff'], metrics['payment_dropoff']):.1f}%", "⚠️ Needs attention", False)
with col5:
    lost_users = metrics["home_users"] - metrics["confirm_users"]
    render_metric_card("Users Lost", f"{lost_users:,}", f"{lost_users/metrics['home_users']*100:.1f}% of total", False)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================
# KEY INSIGHTS SECTION
# ==================================
st.markdown('<div class="section-header">🎯 Key Insights & Recommendations</div>', unsafe_allow_html=True)

insight_cols = st.columns(2)
for i, insight in enumerate(insights[:4]):
    with insight_cols[i % 2]:
        render_insight_box(insight)

# ==================================
# TABS
# ==================================
tab_funnel, tab_segments, tab_trends, tab_deep_dive, tab_data = st.tabs([
    "🔄 Funnel Analysis",
    "👥 Segment Deep-Dive",
    "📈 Trends & Cohorts",
    "🔬 Advanced Analytics",
    "📋 Raw Data",
])

# ==================================
# FUNNEL TAB
# ==================================
with tab_funnel:
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(create_funnel_chart(metrics), use_container_width=True)

    with col_right:
        st.plotly_chart(create_dropoff_waterfall(metrics), use_container_width=True)

    st.markdown('<div class="section-header">Stage-by-Stage Conversion Rates</div>', unsafe_allow_html=True)

    stage_cols = st.columns(3)
    stages = [
        ("Home → Search", metrics["home_to_search_rate"], metrics["home_dropoff"]),
        ("Search → Payment", metrics["search_to_payment_rate"], metrics["search_dropoff"]),
        ("Payment → Confirm", metrics["payment_to_confirm_rate"], metrics["payment_dropoff"]),
    ]

    for col, (stage, rate, dropoff) in zip(stage_cols, stages):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{stage}</div>
                <div class="metric-value" style="color: {'#3fb950' if rate > 20 else '#f85149'}">{rate:.1f}%</div>
                <span class="metric-delta-negative">↓ {dropoff:.1f}% drop-off</span>
            </div>
            """, unsafe_allow_html=True)

# ==================================
# SEGMENTS TAB
# ==================================
with tab_segments:
    segment_choice = st.selectbox(
        "Select Segment",
        ["device", "sex"],
        format_func=lambda x: "📱 Device Type" if x == "device" else "👤 Gender",
    )

    seg_df = segment_analysis[segment_choice]

    st.plotly_chart(create_segment_comparison(seg_df, segment_choice), use_container_width=True)

    st.markdown('<div class="section-header">Detailed Segment Metrics</div>', unsafe_allow_html=True)

    # Format the dataframe for display
    display_df = seg_df[[segment_choice, "funnel_entered", "converted", "conversion_rate", "vs_average", "significant"]].copy()
    display_df.columns = [segment_choice.title(), "Users", "Converted", "Conv. Rate %", "vs Avg %", "Significant"]

    st.dataframe(
        display_df.style.format({
            "Users": "{:,}",
            "Converted": "{:,}",
            "Conv. Rate %": "{:.2f}%",
            "vs Avg %": "{:+.1f}%",
        }).applymap(
            lambda x: "background-color: #1e4620" if x == True else ("background-color: #3d2a14" if x == False else ""),
            subset=["Significant"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("*Significant = p-value < 0.05 (statistically meaningful difference from average)")

    # Cross-segment analysis
    st.markdown('<div class="section-header">Cross-Segment Analysis</div>', unsafe_allow_html=True)

    st.dataframe(
        dropoff_df.style.format({
            "users": "{:,}",
            "conversion_rate": "{:.2f}%",
            "biggest_dropoff_rate": "{:.1f}%",
            "home_dropoff": "{:.1f}%",
            "search_dropoff": "{:.1f}%",
            "payment_dropoff": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ==================================
# TRENDS TAB
# ==================================
with tab_trends:
    st.plotly_chart(create_trend_chart(cohort_df), use_container_width=True)

    st.markdown('<div class="section-header">Weekly Cohort Heatmap</div>', unsafe_allow_html=True)
    st.plotly_chart(create_cohort_heatmap(cohort_df), use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Day of Week Performance</div>', unsafe_allow_html=True)

        fig_dow = px.bar(
            day_analysis,
            x="day",
            y="conversion_rate",
            color="conversion_rate",
            color_continuous_scale=["#f85149", "#f59e0b", "#3fb950"],
            text=[f"{v:.2f}%" for v in day_analysis["conversion_rate"]],
        )
        fig_dow.update_traces(textposition="outside")
        fig_dow.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#f0f2f6"),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(gridcolor="#21262d", title="Conversion Rate (%)"),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Cohort Details</div>', unsafe_allow_html=True)

        st.dataframe(
            cohort_df.style.format({
                "users": "{:,}",
                "conversion_rate": "{:.2f}%",
                "home_to_search": "{:.1f}%",
                "search_to_payment": "{:.1f}%",
                "payment_to_confirm": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
            height=350,
        )

# ==================================
# DEEP DIVE TAB
# ==================================
with tab_deep_dive:
    st.markdown('<div class="section-header">🔬 Statistical Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Device Performance Test")
        device_df = segment_analysis["device"]
        if len(device_df) >= 2:
            best_device = device_df.iloc[0]
            for _, row in device_df.iterrows():
                sig_text = "✅ Statistically significant" if row["significant"] else "⚪ Not significant"
                st.markdown(f"""
                **{row['device']}**: {row['conversion_rate']:.2f}% conversion  
                p-value: {row['p_value']:.4f} | {sig_text}
                """)

    with col2:
        st.markdown("#### Gender Performance Test")
        sex_df = segment_analysis["sex"]
        for _, row in sex_df.iterrows():
            sig_text = "✅ Statistically significant" if row["significant"] else "⚪ Not significant"
            st.markdown(f"""
            **{row['sex']}**: {row['conversion_rate']:.2f}% conversion  
            p-value: {row['p_value']:.4f} | {sig_text}
            """)

    st.markdown('<div class="section-header">📊 Funnel Efficiency Metrics</div>', unsafe_allow_html=True)

    efficiency_cols = st.columns(4)

    with efficiency_cols[0]:
        # Velocity: How fast do users convert (proxy: conversion rate)
        velocity = metrics["overall_conversion"]
        st.metric("Funnel Velocity", f"{velocity:.2f}%", help="Overall conversion rate")

    with efficiency_cols[1]:
        # Leakage: Total users lost
        leakage = (1 - metrics["confirm_users"] / metrics["home_users"]) * 100
        st.metric("Funnel Leakage", f"{leakage:.1f}%", help="Percentage of users who don't convert")

    with efficiency_cols[2]:
        # Bottleneck score (worst stage)
        bottleneck = max(metrics["home_dropoff"], metrics["search_dropoff"], metrics["payment_dropoff"])
        st.metric("Bottleneck Severity", f"{bottleneck:.1f}%", help="Worst drop-off rate")

    with efficiency_cols[3]:
        # Recovery potential
        recovery = metrics["payment_users"] - metrics["confirm_users"]
        st.metric("Cart Recovery Potential", f"{recovery:,}", help="Users who reached payment but didn't convert")

    st.markdown('<div class="section-header">🎯 Optimization Priorities</div>', unsafe_allow_html=True)

    # Calculate impact scores
    priorities = []

    # Impact of fixing each stage
    if metrics["home_dropoff"] > 0:
        impact_search = (metrics["home_users"] - metrics["search_users"]) * 0.1 * (metrics["search_to_payment_rate"]/100) * (metrics["payment_to_confirm_rate"]/100)
        priorities.append(("Improve Home → Search", impact_search, metrics["home_dropoff"]))

    if metrics["search_dropoff"] > 0:
        impact_payment = (metrics["search_users"] - metrics["payment_users"]) * 0.1 * (metrics["payment_to_confirm_rate"]/100)
        priorities.append(("Improve Search → Payment", impact_payment, metrics["search_dropoff"]))

    if metrics["payment_dropoff"] > 0:
        impact_confirm = (metrics["payment_users"] - metrics["confirm_users"]) * 0.1
        priorities.append(("Improve Payment → Confirm", impact_confirm, metrics["payment_dropoff"]))

    priorities.sort(key=lambda x: x[1], reverse=True)

    for i, (action, impact, dropoff) in enumerate(priorities, 1):
        st.markdown(f"""
        **{i}. {action}**  
        - Current drop-off: {dropoff:.1f}%  
        - Potential gain (10% improvement): +{int(impact)} additional conversions
        """)

# ==================================
# RAW DATA TAB
# ==================================
with tab_data:
    st.markdown("### User Funnel Data")
    st.dataframe(
        user_funnel[["user_id", "date", "device", "sex", "stage_home", "stage_search", "stage_payment", "stage_confirm", "converted"]].head(100),
        use_container_width=True,
    )

    with st.expander("View Source Tables"):
        st.markdown("**user_table.csv**")
        st.dataframe(tables["user"].head(20), use_container_width=True)

        st.markdown("**home_page_table.csv**")
        st.dataframe(tables["home"].head(20), use_container_width=True)

        st.markdown("**search_page_table.csv**")
        st.dataframe(tables["search"].head(20), use_container_width=True)

        st.markdown("**payment_page_table.csv**")
        st.dataframe(tables["payment"].head(20), use_container_width=True)

        st.markdown("**payment_confirmation_table.csv**")
        st.dataframe(tables["confirm"].head(20), use_container_width=True)

# ==================================
# FOOTER
# ==================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.85rem; padding: 20px 0;">
    <p>Personal project @Rafli Ardiansyah</p>
</div>
""", unsafe_allow_html=True)
