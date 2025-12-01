# 📊 Sales Funnel Analytics Dashboard

An interactive Streamlit dashboard for analyzing and optimizing sales funnel performance. Built to provide actionable insights on user conversion behavior, segment analysis, and identify optimization opportunities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### Funnel Analysis
- **Conversion Funnel Visualization** — Interactive funnel chart showing user progression through stages
- **Drop-off Waterfall** — Identify exactly where users are leaving the funnel
- **Stage-by-Stage Metrics** — Detailed conversion rates between each funnel stage

### Segment Deep-Dive
- **Device Analysis** — Compare conversion rates across Desktop, Mobile, and Tablet
- **Gender Analysis** — Understand how different demographics convert
- **Cross-Segment Analysis** — Find the best and worst performing user segments
- **Statistical Significance Testing** — Chi-square tests to validate segment differences

### Trends & Cohorts
- **Weekly Cohort Analysis** — Track conversion performance over time
- **Trend Detection** — Automatic identification of improving or declining trends
- **Moving Averages** — Smooth out noise to see the real trajectory
- **Day of Week Performance** — Optimize for the best converting days

### Advanced Analytics
- **Funnel Efficiency Metrics** — Velocity, leakage, and bottleneck scores
- **Optimization Priorities** — Data-driven recommendations ranked by potential impact
- **Revenue Opportunity Calculator** — Estimate potential gains from improvements

### Smart Insights
- Automatically generated actionable recommendations
- Highlights critical drop-off points
- Identifies statistically significant segment differences
- Calculates potential revenue impact

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd funnel
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy plotly scipy
   ```

   Or using uv:
   ```bash
   uv pip install streamlit pandas numpy plotly scipy
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
funnel/
├── app.py                 # Main Streamlit application
├── data/
│   ├── user_table.csv              # User demographics and signup dates
│   ├── home_page_table.csv         # Users who visited the home page
│   ├── search_page_table.csv       # Users who performed a search
│   ├── payment_page_table.csv      # Users who reached payment
│   └── payment_confirmation_table.csv  # Users who completed payment
├── .gitignore
└── README.md
```

## 📊 Data Schema

### user_table.csv
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| date | datetime | User registration/visit date |
| device | string | Device type (Desktop/Mobile/Tablet) |
| sex | string | User gender |

### Funnel Stage Tables
Each funnel stage table contains:
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | User identifier (links to user_table) |

## 🎨 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🔄 Funnel Analysis | Core funnel visualization and drop-off analysis |
| 👥 Segment Deep-Dive | Device and gender segment comparisons |
| 📈 Trends & Cohorts | Time-based analysis and cohort tracking |
| 🔬 Advanced Analytics | Statistical tests and optimization priorities |
| 📋 Raw Data | Explore the underlying data tables |

## 🛠️ Technologies

- **[Streamlit](https://streamlit.io/)** — Web application framework
- **[Pandas](https://pandas.pydata.org/)** — Data manipulation and analysis
- **[Plotly](https://plotly.com/)** — Interactive visualizations
- **[NumPy](https://numpy.org/)** — Numerical computing
- **[SciPy](https://scipy.org/)** — Statistical analysis (chi-square tests)

## 📈 Use Cases

- **E-commerce Optimization** — Identify checkout abandonment issues
- **SaaS Onboarding** — Track user activation funnel
- **Marketing Analysis** — Measure campaign conversion effectiveness
- **Product Analytics** — Understand user behavior patterns

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Rafli Ardiansyah**

---

<p align="center">
  Made with ❤️ using Streamlit
</p>

