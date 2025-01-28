import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from urllib.parse import quote
import torch
import json
import torch.nn.functional as F
import joblib
import torch.nn as nn

css = """
    .stMainBlockContainer {
        padding-top: 50px !important;
    }

    """
st.html(f"<style>{css}</style>")

# @st.cache_data
# def loadData():
#     df = pd.read_csv("../final_prepared_data_with_new.csv")
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Najpierw konwersja do datetime
#     df["date"] = df["date"].astype(str)
#     df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
#     df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
#     df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
#     df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
#     df["round"] = df["round"].astype(int)
#     df["home_goals"] = df["home_goals"].astype(int)
#     df["away_goals"] = df["away_goals"].astype(int)
#     df = df.sort_values("round")
#     dfPL = df[df["league"] == "pl"]
#     dfLL = df[df["league"] == "ll"]
#     dfL1 = df[df["league"] == "l1"]
#     dfBun = df[df["league"] == "bl"]
#     dfSA = df[df["league"] == "sa"]

#     players_pl = pd.read_csv("../players_pl.csv")
#     players_bl = pd.read_csv("../players_bl.csv")
#     players_ll = pd.read_csv("../players_ll.csv")
#     players_l1 = pd.read_csv("../players_l1.csv")
#     players_sa = pd.read_csv("../players_sa.csv")
#     players_new = pd.read_csv("../new_players.csv")
    
#     players = pd.concat([players_pl, players_ll, players_l1, players_bl, players_sa, players_new], ignore_index=True)
#     players = players.rename(columns={"position": "position_x"})
#     players["date"] = pd.to_datetime(players["date"], errors="coerce")  # Najpierw konwersja do datetime
#     players["date"] = players["date"].astype(str)

#     standings = pd.read_csv("../standings_with_new.csv")
#     standings['date']=pd.to_datetime(standings['date'])
#     standings['goal_difference'] = standings['goal_difference'].astype(int)
#     standings['goals'] = standings['goals'].astype(int) 
#     standings['goals_conceded'] = standings['goals_conceded'].astype(int)
#     standingsPL = standings[standings["league"] == "pl"]
#     standingsLL = standings[standings["league"] == "ll"]
#     standingsL1 = standings[standings["league"] == "l1"]
#     standingsBun = standings[standings["league"] == "bl"]
#     standingsSA = standings[standings["league"] == "sa"]

#     odds = pd.read_csv("../odds.csv")
#     oddsPL = odds[odds["Div"] == "E0"]
#     oddsLL = odds[odds["Div"] == "SP1"]
#     oddsL1 = odds[odds["Div"] == "F1"]
#     oddsBun = odds[odds["Div"] == "D1"]
#     oddsSA = odds[odds["Div"] == "I1"]

#     return dfPL, dfLL, dfL1, dfBun, dfSA, standingsPL, standingsLL, standingsL1, standingsBun, standingsSA, players, oddsPL, oddsLL, oddsL1, oddsBun, oddsSA

# # Sprawdzenie, czy dane są już w session_state
# if "dfPL" not in st.session_state:
#     dfPL, dfLL, dfL1, dfBun, dfSA, standingsPL, standingsLL, standingsL1, standingsBun, standingsSA, players, oddsPL, oddsLL, oddsL1, oddsBun, oddsSA = loadData()
#     st.session_state["dfPL"] = dfPL
#     st.session_state["standingsPL"] = standingsPL
#     st.session_state["playersPL"] = players
#     st.session_state["oddsPL"] = oddsPL

#     st.session_state["dfLL"] = dfLL
#     st.session_state["standingsLL"] = standingsLL
#     #st.session_state["playersPL"] = players
#     st.session_state["oddsLL"] = oddsPL

#     st.session_state["dfL1"] = dfL1
#     st.session_state["standingsL1"] = standingsL1
#     #st.session_state["playersPL"] = players
#     st.session_state["oddsL1"] = oddsL1

#     st.session_state["dfBun"] = dfBun
#     st.session_state["standingsBun"] = standingsBun
#     #st.session_state["playersPL"] = players
#     st.session_state["oddsBun"] = oddsBun

#     st.session_state["dfSA"] = dfSA
#     st.session_state["standingsSA"] = standingsSA
#     #st.session_state["playersPL"] = players
#     st.session_state["oddsSA"] = oddsSA

@st.cache_data
def loadData():
    df = pd.read_csv("../data/final_prepared_data_with_new.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Najpierw konwersja do datetime
    df["date"] = df["date"].astype(str)
    df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["round"] = df["round"].astype(int)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.sort_values("round")
    dfPL = df[df["league"] == "pl"]
    dfLL = df[df["league"] == "ll"]
    dfBL = df[df["league"] == "bl"]
    dfL1 = df[df["league"] == "l1"]
    dfSA = df[df["league"] == "sa"]

    df = pd.read_csv("../data/new_matches_fbref.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Najpierw konwersja do datetime
    df["date"] = df["date"].astype(str)
    df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["round"] = df["round"].astype(int)
    df = df.sort_values("round")
    dfPLNew = df[df["league"] == "pl"]
    dfBLNew = df[df["league"] == "bl"]
    dfLLNew = df[df["league"] == "ll"]
    dfL1New = df[df["league"] == "l1"]
    dfSANew = df[df["league"] == "sa"]

    standings = pd.read_csv("../data/standings_with_new.csv")
    standings['date']=pd.to_datetime(standings['date'])
    standings['goal_difference'] = standings['goal_difference'].astype(int)
    standings['goals'] = standings['goals'].astype(int) 
    standings['goals_conceded'] = standings['goals_conceded'].astype(int)
    standingsPL = standings[standings["league"] == "pl"]
    standingsBL = standings[standings["league"] == "bl"]
    standingsLL = standings[standings["league"] == "ll"]
    standingsL1 = standings[standings["league"] == "l1"]
    standingsSA = standings[standings["league"] == "sa"]

    return dfPL, dfLL, dfBL, dfL1, dfSA, standingsPL, standingsLL, standingsBL, standingsL1, standingsSA, dfPLNew, dfBLNew, dfLLNew, dfL1New, dfSANew

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler

def load_selected_fetures(selected_features_path):
    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    return selected_features

def predict_outcome(input_features, model):
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        prediction = model(input_tensor)
        return prediction.squeeze()[0].item(), prediction.squeeze()[1].item(), prediction.squeeze()[2].item()
    
def getCourse(prob):
    return round(1 / prob, 2)

def generate_html_table(teams_stats):
    html_template = """
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
                text-align: center;
                background-color: #f9f9f9;
                color:black;
                border: 0px solid rgba(34, 34, 38, 0.25);
                font: Arial;
            th, td {{
                padding: 5px;
                border: 0px solid rgba(34, 34, 38, 0.25);
                text-align: center;
                width: 1%;
            }}
            th {{
                background-color: #f9f9f9;
                color: rgba(34, 34, 38, 0.45);
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #e6e6e6;
            }}
            td {{
                line-height: 25px;
                padding-top: 2px;
                padding-bottom: 2px;
            }}
            th:last-child, td:last-child {{
                font-weight: bold;
            }}
            th:nth-child(2), td:nth-child(2) {{
                width: 15%;
                text-align: left;
            }}
            th:nth-child(1), td:nth-child(1) {{
                width: 0%;
            }}
            .highlight-green td:nth-child(1) span {{
                display: inline-block;
                width: 25px;
                height: 25px;
                line-height: 25px;
                border-radius: 50%;
                background-color: #26943b;
                color: white;
                font-weight: bold;
            }}
            .highlight-red td:nth-child(1) span {{
                display: inline-block;
                width: 25px;
                height: 25px;
                line-height: 25px;
                border-radius: 50%;
                background-color: #c1262d;
                color: white;
                font-weight: bold;
            }}
        </style>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Zespół</th>
                    <th>M</th>
                    <th>W</th>
                    <th>R</th>
                    <th>P</th>
                    <th>+/-</th>
                    <th>+</th>
                    <th>-</th>
                    <th>Pkt</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    """

    rows = ""
    for team in teams_stats:
        row_class = (
            "highlight-green" if team["highlight"] == "green" else "highlight-red" if team["highlight"] == "red" else ""
        )
        rows += f"""
        <tr class="{row_class}">
            <td><span>{team["position"]}</span></td>
            <td>{team["name"]}</td>
            <td>{team["played"]}</td>
            <td>{team["wins"]}</td>
            <td>{team["draws"]}</td>
            <td>{team["losses"]}</td>
            <td>{team["diff"]}</td>
            <td>{team["goals_scored"]}</td>
            <td>{team["goals_conceded"]}</td>
            <td>{team["points"]}</td>
        </tr>
        """
    return html_template.format(rows=rows)

def generate_html_match_list(df, league):
    scaler_outcome = load_scaler("./models/outcome_scaler.pkl")
    selected_features_outcome = load_selected_fetures("./models/outcome_features.json")
    model_outcome = load_model("./models/football_match_predictor_v1.pth")
    html_template = """
        <style>
            .container {
                max-width: 800px;
                margin: 20px auto;
                padding: 10px;
            }
            .round {
                font-size: 18px;
                margin-bottom: 20px;
                margin-top: 20px;
                text-align: left;
                font-weight: bold;
                background-color: #eee;
                color: grey;
                border-radius: 6px;
                padding-left: 12px;
                padding-top: 5px;
                padding-bottom: 5px;
            }
            .match {
                display: grid;
                grid-template-columns: 1.2fr 1.9fr 0.2fr 3.8fr;
                align-items: center;
                background-color: white;
                border-radius: 8px;
                margin-bottom: 8px;
                padding: 5px 10px;
                color: black;
            }
            .match:hover {
                background-color: #e6e6e6;
            }
            .time-date {
                font-size: 16px;
                color: rgba(12, 12, 12, 0.65);
            }
            .teams {
                display: flex;
                flex-direction: column;
                justify-content: center;
                font-size: 16px;
            }
            .winner {
                font-weight: bold;
            }
            .win {
                background-color: #26943b !important;
            }
            .away-team {
                margin-top: 5px;
            }
            .score {
                text-align: right;
                font-weight: bold;
                font-size: 18px;
            }
            .cell {
                display: inline-block;
                width: 25%;
                height: 40px;
                background-color: white;
                border-radius: 6px;
                border: 1px solid #eee;
                padding: 0 10px;
                color: black;
                font-family: Arial, sans-serif;
                line-height: 40px;
                margin-left: 6px;
                margin-right: 4px;
            }
            .result {
                font-size: 14px;
                text-align: left;
                font-weight: 0;
                float: left;
            }
            .odds {
                font-size: 16px;
                font-weight: bold;
                text-align: right;
            }
            hr {
                width: 100%;
                color: #eee;
                margin: 0;
            }
            a {
                text-decoration: none !important;
            }
            a:hover {
                text-decoration: none !important;
            }
        </style>
    """

    html_template += """<div class="container">"""


    match_template = """
    <a href="/Statystyki_Przedmeczowe?home_team={encoded_home_team}&date={original_date}&league={league}" target=_self>
    <div class="match">
        <div class="time-date">{date}  {time}</div>
        <div class="teams">
            <div class="home-team{home_class}">{home_team}</div>
            <div class="away-team{away_class}">{away_team}</div>
        </div>
        <div class="score">
            <div>{home_goals}</div>
            <div>{away_goals}</div>
        </div>
        <div class="odds">
            <div class="cell{home_course}">
                <span class="result">1</span>
                <span class="odds">{home_win_course:.2f}</span>
            </div>
            <div class="cell{drawing_course}">
                <span class="result">X</span>
                <span class="odds">{draw_course:.2f}</span>
            </div>
            <div class="cell{away_course}">
                <span class="result">2</span>
                <span class="odds">{away_win_course:.2f}</span>
            </div>
        </div>
    </div></a>
    <hr>
    """

    for roundi in df["round"].unique():
        matches_html = ""
        for _, row in df[df["round"] == roundi].iterrows():
            filtered_matches = df[(df["date"] == row["date"]) & (df["home_team"] == row["home_team"])]
            filtered_matches = filtered_matches[[col for col in df.columns if 'last5' in col or 'matches_since' in col or 'overall' in col or 'tiredness' in col]]
            filtered_matches = filtered_matches.drop(columns = ["home_last5_possession", "away_last5_possession"])
            all_features = filtered_matches.iloc[0]
            expected_order = scaler_outcome.feature_names_in_
            all_features = all_features.reindex(expected_order)
            filtered_matches = filtered_matches.reindex(columns=expected_order)
            all_features_scaled_outcome = scaler_outcome.transform([all_features])
            input_features_outcome = all_features_scaled_outcome[:, [filtered_matches.columns.get_loc(col) for col in selected_features_outcome]]
            draw, home_win, away_win = predict_outcome(input_features_outcome, model_outcome)
            home_class = ""
            away_class = ""
            home_course = ""
            draw_course = ""
            away_course = ""
            if row["new"]:
                row["home_goals"] = "-"
                row["away_goals"] = "-"
            else:
                row["home_goals"] = int(row["home_goals"])
                row["away_goals"] = int(row["away_goals"])
                if row["home_goals"] > row["away_goals"]:
                    home_class = " winner"
                    home_course = " win"
                elif row["home_goals"] < row["away_goals"]:
                    away_class = " winner"
                    away_course = " win"
                else:
                    draw_course = " win"

            matches_html += match_template.format(
                date=row["date"][-2:]+"."+row["date"][5:7],
                original_date = row["date"],
                time=row["time"],
                home_team=row["home_team"],
                encoded_home_team = quote(row["home_team"]),
                away_team=row["away_team"],
                home_goals=row["home_goals"],
                away_goals=row["away_goals"],
                home_win_course=getCourse(home_win),
                draw_course=getCourse(draw),
                away_win_course=getCourse(away_win),
                home_class = home_class,
                away_class = away_class,
                home_course = home_course,
                away_course = away_course,
                drawing_course = draw_course,
                league = league
            )
        html_template += f"""<div class="round">Kolejka {roundi}</div>
        {matches_html}"""
        

    # Format the final HTML
    html_template += "</div>"

    return html_template

def generate_league_table(standings, place, league):
    current_standings_date = sorted(standings['date'].unique())[-1]
    standings = standings[standings["date"] == current_standings_date]
    selected_columns_standings = ['team', 'matches_played', 'wins', 'draws', 'defeats', 'goal_difference', 'goals', 'goals_conceded', 'points']
    table = standings[selected_columns_standings]
    table = table.sort_values(["points", "goal_difference", "goals"], ascending=False)
    table['place'] = range(1, len(table) + 1)
    table = table.set_index('place')
    standings_data = []
    for i, row in table.iterrows():
        if i<=5 or i>=place:
            team_stats = {}
            team_stats["position"] = i
            team_stats["highlight"] = ""
            if i<5:
                team_stats["highlight"] = "green"
            if i>place+1:
                team_stats["highlight"] = "red"
            team_stats["name"] = row['team']
            team_stats["played"] = row['matches_played']
            team_stats["wins"] = row['wins']
            team_stats["draws"] = row['draws']
            team_stats["losses"] = row['defeats']
            team_stats["diff"] = row['goal_difference']
            if team_stats["diff"] > 0:
                team_stats["diff"] = "+" + str(row['goal_difference'])
            team_stats["goals_scored"] = row['goals']
            team_stats["goals_conceded"] = row['goals_conceded']
            team_stats["points"] = row['points']
            standings_data.append(team_stats)
        if i==6:
            team_stats = {}
            team_stats["position"] = "..."
            team_stats["highlight"] = ""
            team_stats["name"] = "..."
            team_stats["played"] = "..."
            team_stats["wins"] = "..."
            team_stats["draws"] = "..."
            team_stats["losses"] = "..."
            team_stats["diff"] = "..."
            team_stats["goals_scored"] = "..."
            team_stats["goals_conceded"] = "..."
            team_stats["points"] = "..."
            standings_data.append(team_stats)

    html_table = generate_html_table(standings_data)
    html_table_final = """
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: "Source Sans Pro", sans-serif;
            margin: 0;
        }
        .tab {
            box-sizing: border-box;
            border: 1px solid #ddd;
            border-radius: 8px;
            width: 95%;
            padding: 18px 14px 16px 14px;
            background-color: #f9f9f9;
            margin: auto;
            margin-top: 18px;
        }
        .tab_title {
            font-size: 22px;
            font-weight: bold;
            color: #333;
            width: 100%;
            text-align: center;
            margin-bottom: 12px;
        }
    </style> """ + f"""
    <div class="tab">
    <div class="tab_title">{league}</div>
    {html_table}
    </div>
    """
    return html_table_final


dfPL, dfLL, dfBL, dfL1, dfSA, standingsPL, standingsLL, standingsBL, standingsL1, standingsSA, dfPLNew, dfBLNew, dfLLNew, dfL1New, dfSANew = loadData()

st.markdown(
    """
    <a href="/Premier_League" target=_self style="
        text-decoration: none; 
        color: black; 
        display: inline-block; 
        font-size: 32px; 
        font-weight: bold; 
        padding: 20px 40px; 
        background-color: #f0f0f0; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        ">
        Premier League
    </a>
    """,
    unsafe_allow_html=True
)

if len(dfPLNew["round"])>0:
    records_to_show = dfPLNew.sort_values(by = ['date', 'time']).head(3)
    records_to_show["new"] = True
else:
    records_to_show = dfPL.sort_values(by = ['date', 'time'], ascending=False).head(3)
    records_to_show["new"] = False

html_table_final = generate_league_table(standingsPL, 16, "Premier League")

col1, col2 = st.columns([3,2])
with col1:
    st.markdown(generate_html_match_list(records_to_show, "pl"), unsafe_allow_html=True)
    # st.components.v1.html(weather_style, height=190)
    # st.markdown(html_h2h, unsafe_allow_html=True)
    # st.components.v1.html(form_html, height=240)
    # st.pyplot(fig21)
    # if (len(probabilities)>0):
    #     st.pyplot(fig22)
with col2:
    #st.markdown(tab_html, unsafe_allow_html=True)
    st.components.v1.html(html_table_final, height=520)

st.markdown(
    """
    <a href="/La_Liga" target=_self style="
        text-decoration: none; 
        color: black; 
        display: inline-block; 
        font-size: 32px; 
        font-weight: bold; 
        padding: 20px 40px; 
        background-color: #f0f0f0; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        ">
        La Liga
    </a>
    """,
    unsafe_allow_html=True
)

if len(dfLLNew["round"])>0:
    records_to_show = dfLLNew.sort_values(by = ['date', 'time']).head(3)
    records_to_show["new"] = True
else:
    records_to_show = dfLL.sort_values(by = ['date', 'time'], ascending=False).head(3)
    records_to_show["new"] = False

html_table_final = generate_league_table(standingsLL, 16, "La Liga")

col1, col2 = st.columns([3,2])
with col1:
    st.markdown(generate_html_match_list(records_to_show, "ll"), unsafe_allow_html=True)
    # st.components.v1.html(weather_style, height=190)
    # st.markdown(html_h2h, unsafe_allow_html=True)
    # st.components.v1.html(form_html, height=240)
    # st.pyplot(fig21)
    # if (len(probabilities)>0):
    #     st.pyplot(fig22)
with col2:
    #st.markdown(tab_html, unsafe_allow_html=True)
    st.components.v1.html(html_table_final, height=490)


st.markdown(
    """
    <a href="/Ligue_1" target=_self style="
        text-decoration: none; 
        color: black; 
        display: inline-block; 
        font-size: 32px; 
        font-weight: bold; 
        padding: 20px 40px; 
        background-color: #f0f0f0; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        ">
        Ligue 1
    </a>
    """,
    unsafe_allow_html=True
)

if len(dfL1New["round"])>0:
    records_to_show = dfL1New.sort_values(by = ['date', 'time']).head(3)
    records_to_show["new"] = True
else:
    records_to_show = dfL1.sort_values(by = ['date', 'time'], ascending=False).head(3)
    records_to_show["new"] = False

html_table_final = generate_league_table(standingsL1, 14, "Ligue 1")

col1, col2 = st.columns([3,2])
with col1:
    st.markdown(generate_html_match_list(records_to_show, "l1"), unsafe_allow_html=True)
    # st.components.v1.html(weather_style, height=190)
    # st.markdown(html_h2h, unsafe_allow_html=True)
    # st.components.v1.html(form_html, height=240)
    # st.pyplot(fig21)
    # if (len(probabilities)>0):
    #     st.pyplot(fig22)
with col2:
    #st.markdown(tab_html, unsafe_allow_html=True)
    st.components.v1.html(html_table_final, height=490)


st.markdown(
    """
    <a href="/Bundesliga" target=_self style="
        text-decoration: none; 
        color: black; 
        display: inline-block; 
        font-size: 32px; 
        font-weight: bold; 
        padding: 20px 40px; 
        background-color: #f0f0f0; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        ">
        Bundesliga
    </a>
    """,
    unsafe_allow_html=True
)

if len(dfBLNew["round"])>0:
    records_to_show = dfBLNew.sort_values(by = ['date', 'time']).head(3)
    records_to_show["new"] = True
else:
    records_to_show = dfBL.sort_values(by = ['date', 'time'], ascending=False).head(3)
    records_to_show["new"] = False

html_table_final = generate_league_table(standingsBL, 14, "Bundesliga")

col1, col2 = st.columns([3,2])
with col1:
    st.markdown(generate_html_match_list(records_to_show, "bl"), unsafe_allow_html=True)
    # st.components.v1.html(weather_style, height=190)
    # st.markdown(html_h2h, unsafe_allow_html=True)
    # st.components.v1.html(form_html, height=240)
    # st.pyplot(fig21)
    # if (len(probabilities)>0):
    #     st.pyplot(fig22)
with col2:
    #st.markdown(tab_html, unsafe_allow_html=True)
    st.components.v1.html(html_table_final, height=490)

st.markdown(
    """
    <a href="/Serie_A" target=_self style="
        text-decoration: none; 
        color: black; 
        display: inline-block; 
        font-size: 32px; 
        font-weight: bold; 
        padding: 20px 40px; 
        background-color: #f0f0f0; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        ">
        Serie A
    </a>
    """,
    unsafe_allow_html=True
)

if len(dfSANew["round"])>0:
    records_to_show = dfSANew.sort_values(by = ['date', 'time']).head(3)
    records_to_show["new"] = True
else:
    records_to_show = dfSA.sort_values(by = ['date', 'time'], ascending=False).head(3)
    records_to_show["new"] = False

html_table_final = generate_league_table(standingsSA, 16, "Serie A")

col1, col2 = st.columns([3,2])
with col1:
    st.markdown(generate_html_match_list(records_to_show, "sa"), unsafe_allow_html=True)
    # st.components.v1.html(weather_style, height=190)
    # st.markdown(html_h2h, unsafe_allow_html=True)
    # st.components.v1.html(form_html, height=240)
    # st.pyplot(fig21)
    # if (len(probabilities)>0):
    #     st.pyplot(fig22)
with col2:
    #st.markdown(tab_html, unsafe_allow_html=True)
    st.components.v1.html(html_table_final, height=490)

st.markdown(
    """
    <a href="/Your_model" target=_self style="
        text-decoration: none; 
        color: black; 
        display: inline-block; 
        font-size: 32px; 
        font-weight: bold; 
        padding: 20px 40px; 
        background-color: #f0f0f0; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        ">
        Stwórz swój własny model
    </a>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="margin: 50px 0;">
        <hr style="border: none; height: 2px; background-color: black; margin: 0; width: 100%;">
    </div>
    """,
    unsafe_allow_html=True
)

# Define local image paths and text content
# columns_data = [
#     {"image_path": "graphics/Germany.png", "text": "Bundesliga", "button_label": "Przejdź do strony"},
#     {"image_path": "graphics/Spain.png", "text": "La Liga", "button_label": "Przejdź do strony"},
#     {"image_path": "graphics/France.png", "text": "Ligue 1", "button_label": "Przejdź do strony"},
#     {"image_path": "graphics/England.png", "text": "Premier League", "button_label": "Przejdź do strony"},
#     {"image_path": "graphics/Italy.png", "text": "Serie A", "button_label": "Przejdź do strony"}
# ]

# Create 5 columns
# cols = st.columns(5)

# # Populate each column with content
# for i, col in enumerate(cols):
#     with col:
#         try:
#             image = Image.open(columns_data[i]['image_path'])
#             st.image(image.resize((1280,854)), use_container_width=True)

#         except FileNotFoundError:
#             col.error(f"Nie znaleziono obrazu: {columns_data[i]['image_path']}")

# cols = st.columns(5)
# for i, col in enumerate(cols):
#     with col:
#         if st.button(
#                 f"{columns_data[i]['text']}",
#                 key=f"HomeButton{i}"
#             ):
#                 st.switch_page(f"pagesVis/{columns_data[i]['text']}.py")

st.write("""
Zakłady bukmacherskie to forma rozrywki, która cieszy się dużą popularnością na całym świecie. 
Jednak, podobnie jak każda inna forma hazardu, wiąże się z ryzykiem. Ważne jest, aby zdawać sobie sprawę, że **zakłady bukmacherskie są przeznaczone wyłącznie dla osób pełnoletnich**, które są świadome konsekwencji swojej decyzji.
""")

st.write("""
Gry bukmacherskie mogą być ekscytujące, ale niosą ze sobą także niebezpieczeństwo uzależnienia. 
Wiele osób, które zaczynają od okazjonalnej zabawy, z czasem stają się zależne od tej formy rozrywki, co może prowadzić do poważnych problemów finansowych, emocjonalnych, a także rodzinnych.

Dlatego pamiętaj, żeby:
- Grać odpowiedzialnie i nie przekraczać swoich możliwości finansowych,
- Ustalić limity czasowe oraz finansowe i nie łamać ich,
- Zawsze traktować zakłady bukmacherskie jako formę zabawy, a nie sposób na zarobek,
- Szukać pomocy, jeśli zauważysz, że hazard staje się problemem w Twoim życiu.
""")

st.write("""
Uzależnienie od hazardu to poważna choroba, która może dotknąć każdego. Jeśli czujesz, że hazard wpływa na Twoje życie w negatywny sposób, nie wahaj się szukać pomocy. Są organizacje i specjaliści, którzy oferują wsparcie osobom borykającym się z tym problemem.

**Pamiętaj!** Hazard to tylko zabawa, ale tylko wtedy, gdy kontrolujesz, co robisz. Graj z głową i nigdy nie pozwól, by zabawa stała się problemem.
""")

st.write("W przypadku problemów z uzależnieniem, skontaktuj się z lokalnymi organizacjami wsparcia.")