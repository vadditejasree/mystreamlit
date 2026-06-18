
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(page_title="Team Ranking Intelligence Platform", layout="wide")

st.title("🏆 Team Ranking Intelligence Platform")
st.caption("FIFA World Cup — Cross-Tournament Team Rankings & Performance Analytics")

uploaded_file = st.file_uploader("Upload WorldCupMatches.csv", type=["csv"])

@st.cache_data
def compute_team_stats(df):
    records = []

    for _, row in df.iterrows():
        ht, at = row['Home Team Name'], row['Away Team Name']
        hg, ag = int(row['Home Team Goals']), int(row['Away Team Goals'])
        yr = int(row['Year'])

        records.append({
            'Team': ht, 'Year': yr,
            'GF': hg, 'GA': ag,
            'Win': 1 if hg > ag else 0,
            'Draw': 1 if hg == ag else 0,
            'Loss': 1 if hg < ag else 0,
        })

        records.append({
            'Team': at, 'Year': yr,
            'GF': ag, 'GA': hg,
            'Win': 1 if ag > hg else 0,
            'Draw': 1 if hg == ag else 0,
            'Loss': 1 if ag < hg else 0,
        })

    match_df = pd.DataFrame(records)

    stats = match_df.groupby('Team').agg(
        Tournaments=('Year', 'nunique'),
        Matches_Played=('Win', 'count'),
        Wins=('Win', 'sum'),
        Draws=('Draw', 'sum'),
        Losses=('Loss', 'sum'),
        Goals_For=('GF', 'sum'),
        Goals_Against=('GA', 'sum')
    ).reset_index()

    stats['Goal_Difference'] = stats['Goals_For'] - stats['Goals_Against']
    stats['Points'] = stats['Wins'] * 3 + stats['Draws']
    stats['Win_Rate'] = stats['Wins'] / stats['Matches_Played'] * 100
    stats['Goals_Per_Game'] = stats['Goals_For'] / stats['Matches_Played']
    stats['Points_Per_Game'] = stats['Points'] / stats['Matches_Played']

    return stats

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df = df.dropna(subset=[
        'Home Team Goals', 'Away Team Goals',
        'Year', 'Home Team Initials', 'Away Team Initials'
    ])

    team_stats = compute_team_stats(df)

    ranked = team_stats[team_stats['Matches_Played'] >= 10].copy()

    score_cols = [
        'Points', 'Win_Rate', 'Goal_Difference',
        'Goals_Per_Game', 'Tournaments', 'Points_Per_Game'
    ]

    weights = {
        'Points': 0.30,
        'Win_Rate': 0.25,
        'Goal_Difference': 0.20,
        'Goals_Per_Game': 0.10,
        'Tournaments': 0.08,
        'Points_Per_Game': 0.07
    }

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ranked[score_cols].fillna(0))
    scaled_df = pd.DataFrame(scaled, columns=score_cols)

    ranked['Intelligence_Score'] = sum(
        scaled_df[c] * w for c, w in weights.items()
    ) * 100

    ranked = ranked.sort_values(
        'Intelligence_Score', ascending=False
    ).reset_index(drop=True)

    st.subheader("Top Ranked Teams")
    st.dataframe(ranked.head(20), use_container_width=True)

    top15 = ranked.head(15)

    fig = px.bar(
        top15,
        x="Intelligence_Score",
        y="Team",
        orientation="h",
        title="Top 15 Teams by Intelligence Score"
    )
    st.plotly_chart(fig, use_container_width=True)

    scatter = px.scatter(
        ranked,
        x="Win_Rate",
        y="Goal_Difference",
        size="Matches_Played",
        color="Intelligence_Score",
        hover_name="Team",
        title="Win Rate vs Goal Difference"
    )
    st.plotly_chart(scatter, use_container_width=True)

    csv = ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Rankings",
        csv,
        "team_rankings.csv",
        "text/csv"
    )
else:
    st.info("Upload the FIFA World Cup matches CSV file to begin.")
