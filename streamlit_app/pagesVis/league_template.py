import streamlit as st
import numpy as np
import pandas as pd
import runpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import json
import torch.nn.functional as F
import joblib
import torch.nn as nn
from urllib.parse import quote


def loadPage(current_league):
    if current_league == "pl":
        league_name = "Premier League"
    elif current_league == "bl":
        league_name = "Bundesliga"
    elif current_league == "ll":
        league_name = "La Liga"
    elif current_league == "l1":
        league_name = "Ligue 1"
    elif current_league == "sa":
        league_name = "Serie A"
    else:
        league_name = "None of top 5"
        # Dodaj inne komponenty dla Serie A
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    theme = st.session_state.theme

    css = """
    .stMainBlockContainer {
        padding-top: 30px !important;
    }

    """
    st.html(f"<style>{css}</style>")

    # Chowanie statystyk po zmianie filtrów
    def restartStats():
        pass
        # for i in range (st.session_state["PLnumber_of_matches"]):
        #     if f"PLshow_row_{i}" in st.session_state:
        #         st.session_state[f"PLshow_row_{i}"] = False

    # chyba do usunięcia
    def showDateButton():
        if len(season_filter_matches) == 0:
            st.session_state["show_tablePL"] = True
        else:
            st.session_state["show_tablePL"] = False
        restartStats()

    @st.cache_data
    def statsGraph(home_stats, away_stats, categories):
        total_stats = np.array(home_stats) + np.array(away_stats)
        home_ratios = np.array(home_stats) / total_stats
        away_ratios = np.array(away_stats) / total_stats

        fig, ax = plt.subplots(figsize=(8, len(categories) * 0.4))
        ax.set_facecolor("#1A1A1A") 

        for j, (category, home_ratio, away_ratio) in enumerate(zip(categories, home_ratios, away_ratios)):
            y_position = len(categories) - j  # Pozycja w osi Y (odwracamy kolejność)
            home_color = "#003366"
            away_color = "#003366"
            if home_ratio > away_ratio:
                home_color = "#CC0033"
            elif home_ratio < away_ratio:
                away_color = "#CC0033"

            ax.text(0, y_position + 0.36, category, ha="center", va="center", fontsize=10, color="black", weight="bold")
                    
            ax.barh(
                y_position, -home_ratio, height=0.15, color=home_color, align="center", 
                zorder=3, edgecolor="black", linewidth=1.5
            )
            ax.barh(
                y_position, away_ratio, height=0.15, color=away_color, align="center", 
                zorder=3, edgecolor="black", linewidth=1.5
            )
            ax.text(-home_ratio - 0.02, y_position, f"{home_stats[j]}", ha="right", va="center", fontsize=10, color="black")
            ax.text(away_ratio + 0.02, y_position, f"{away_stats[j]}", ha="left", va="center", fontsize=10, color="black")

        ax.set_xlim(-1, 1)  # Oś X od -1 do 1 (po równo na obie strony)
        ax.set_ylim(0.5, len(categories) + 0.5)  # Oś Y dla odpowiedniego rozmieszczenia
        ax.axis("off")  # Usunięcie osi, ponieważ nie są potrzebne

        plt.tight_layout()
        st.pyplot(plt)

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

    def predict_goals(input_features, model):
        with torch.no_grad():
            input_tensor = torch.tensor(input_features, dtype=torch.float32)
            prediction = model(input_tensor)
            return prediction.squeeze()[0].item(), prediction.squeeze()[1].item()
        
    def predict_outcome(input_features, model):
        with torch.no_grad():
            input_tensor = torch.tensor(input_features, dtype=torch.float32)
            prediction = model(input_tensor)
            return prediction.squeeze()[0].item(), prediction.squeeze()[1].item(), prediction.squeeze()[2].item()


    def generate_html_table(teams_stats):
        html_template = """
            <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    text-align: center;
                    background-color: white;
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
                    background-color: white;
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
                    background-color: #28a745;
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
        </body>
        </html>
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

    def getCourse(prob):
        return round(1 / prob, 2)

    def generate_html_match_list(df):
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
                    background-color: #28a745 !important;
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
                    width: 20%;
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
                    margin: 0 !important;
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
        <a href="/{url_start}?home_team={encoded_home_team}&date={original_date}&league={current_league}" target=_self>
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
                    url_start = "Statystyki_Przedmeczowe"
                    row["home_goals"] = "-"
                    row["away_goals"] = "-"
                else:
                    url_start = "Statystyki_Pomeczowe"
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
                    url_start = url_start,
                    current_league=current_league
                )
            html_template += f"""<div class="round">Kolejka {roundi}</div>
            {matches_html}"""
            

        # Format the final HTML
        html_template += "</div>"

        return html_template
    @st.cache_data
    def loadData(current_league):
        df = pd.read_csv("../data/final_prepared_data_with_weather_new.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].astype(str)
        df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
        df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
        df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
        df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
        df["round"] = df["round"].astype(int)
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)
        df = df.sort_values("round")
        dfPL = df[df["league"] == current_league]

        df = pd.read_csv("../data/new_matches_fbref.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].astype(str)
        df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
        df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
        df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
        df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
        df["round"] = df["round"].astype(int)
        df = df.sort_values("round")
        dfPLNew = df[df["league"] == current_league]

        standings = pd.read_csv("../data/standings_with_new.csv")
        standings['date']=pd.to_datetime(standings['date'])
        standings['goal_difference'] = standings['goal_difference'].astype(int)
        standings['goals'] = standings['goals'].astype(int) 
        standings['goals_conceded'] = standings['goals_conceded'].astype(int)
        standingsPL = standings[standings["league"] == current_league]

        return dfPL, dfPLNew, standingsPL


    df, df_new, standings = loadData(current_league)
    df_filtered=df.copy()
    standings_filtered=standings.copy()
    df_filtered_new = df_new.copy()

    # if "PLseason_filter" not in st.session_state:
    #     st.session_state["PLseason_filter"] = []
    # if "PLteam_filter" not in st.session_state:
    #     st.session_state["PLteam_filter"] = []
    # if "PLnumber_of_matches" not in st.session_state:
    #     st.session_state["PLnumber_of_matches"] = 10


    st.title(f"{league_name}")
    col1, col2 = st.columns(2)
    with col1:
        season_filter = st.multiselect("Wybierz sezon, z którego chcesz zobaczyć tabelę oraz statystyki",
            options = standings['season'].unique(), on_change=showDateButton, max_selections=1, key=f"{current_league}_season_filter")

        if season_filter == []:
            season_filter_matches = sorted(standings['season'].unique(), reverse=True)[0]
        else:
            season_filter_matches = season_filter[0]

        date_standings = max(standings_filtered['date'].dt.strftime('%Y-%m-%d'))

    with col2:
        if len(season_filter) == 1:
            standings_filtered = standings[standings['season'] == season_filter_matches]
            date_standings = st.date_input("Wybierz datę tabeli",
                min_value = min(standings_filtered['date']),
                max_value = max(standings_filtered['date']),
                value = max(standings_filtered['date']))

        possible_date = max(standings_filtered[standings_filtered["date"] <= pd.to_datetime(date_standings)]["date"].unique())
        standings_filtered = standings_filtered[standings_filtered["date"] == possible_date]

    # Filtrowanie i wyświetlanie tabeli
    st.subheader(f"Tabela {league_name} w sezonie {season_filter_matches}")
    st.caption(f"Stan na: {date_standings}")
    date_standings = pd.to_datetime(date_standings)

    selected_columns_standings = ['team', 'matches_played', 'wins', 'draws', 'defeats', 'goal_difference', 'goals', 'goals_conceded', 'points']
    table = standings_filtered[selected_columns_standings]
    table = table.sort_values(["points", "goal_difference", "goals"], ascending=False)
    table['place'] = range(1, len(table) + 1)
    table = table.set_index('place')
    standings_data = []
    for i, row in table.iterrows():
        team_stats = {}
        team_stats["position"] = i
        team_stats["highlight"] = ""
        if i<5:
            team_stats["highlight"] = "green"
        if (current_league == 'bl' or current_league == 'l1') and i>15:
            team_stats["highlight"] = "red"
        elif i>17:
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

    html_table = generate_html_table(standings_data)
    # table.columns = [ 'Zespół', 'Mecze rozegrane', 'Wygrane', 'Remisy', 'Porażki', 'Różnica bramek', 'Bramki strzelone', 'Bramki stracone', 'Punkty']

    col1, col2, col3 = st.columns([1,5,1])
    with col2:
        # st.table(table)
        st.components.v1.html(html_table, height=670)

    # Filtry dla meczów
    filtr1, filtr2 = st.columns(2)

    with filtr1:
        team_filter = st.multiselect("Wybierz drużynę", options = sorted(df_filtered[df_filtered['season'] == season_filter_matches]['home_team'].unique()), key=f"{current_league}_team_filter")
    # with filtr2:
    #     number_of_matches = st.slider("Wybierz liczbę wyświetlanych meczów", min_value=10, max_value=100, step=5, value=st.session_state["PLnumber_of_matches"], key="PLnumber_of_matches")

    team_filter2=team_filter


    # Filtrowanie danych

    if team_filter==[]:
        team_filter2=df['home_team'].unique()
    df_filtered=df_filtered[(pd.to_datetime(df['date'])<=date_standings)
                & (df_filtered['season'] == season_filter_matches)
                & ((df_filtered['home_team'].isin(team_filter2))
                    | (df_filtered['away_team'].isin(team_filter2)))]
    df_filtered.sort_values(by=["round", "date", "time"], ascending=False, inplace=True)

    df_filtered_new=df_filtered_new[(df_filtered_new['season'] == season_filter_matches)
                & ((df_filtered_new['home_team'].isin(team_filter2))
                    | (df_filtered_new['away_team'].isin(team_filter2)))]
    df_filtered_new.sort_values(by=["round", "date", "time"], ascending=False, inplace=True)

    # Wypisywanie danych
    if team_filter==[]:
        records_to_show = df_filtered[df_filtered["round"] >= max(df_filtered["round"])-1]
    else:
        records_to_show = df_filtered
    if len(df_filtered_new["round"])>0:
        new_records_to_show = df_filtered_new[df_filtered_new["round"] <= min(df_filtered_new["round"])+1]
    else:
        new_records_to_show = df_filtered_new
    col1, col2, col3 = st.columns([1,7,1])

    new_records_to_show["new"] = True
    records_to_show["new"] = False

    if date_standings == max(standings_filtered['date']):
        all_records_to_show = pd.concat([new_records_to_show, records_to_show])
    else:
        all_records_to_show = records_to_show


    with col2:
        # st.components.v1.html(generate_html_match_list(records_to_show), height=4000)
        st.markdown(generate_html_match_list(all_records_to_show), unsafe_allow_html=True)

    # for i in range(min(50, df_filtered['home_team'].count())):
    #     col1, col2, col3, col4, col5, col6 = st.columns([3,5,2,5,2,2])
    #     with col1:
    #         st.markdown(f"""
    #                 <div style="text-align: center; font-size: 15px;
    #                     background-color: #f8f9ab; 
    #                     padding: 20px 0;
    #                     margin: 10px;
    #                     margin-top: 0;
    #                     box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['date']} {df_filtered.iloc[i]['time']}
    #                 </div>""", unsafe_allow_html=True)
    #     with col2:
    #         st.markdown(f"""
    #                 <div style="text-align: center; font-size: 15px;
    #                     background-color: #f8f9ab; 
    #                     padding: 20px 0;
    #                     margin: 10px;
    #                     margin-top: 0;
    #                     box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_team']}
    #                 </div>""", unsafe_allow_html=True)
    #     with col3:
    #         st.markdown(f"""
    #                 <div style="text-align: center; font-size: 15px;
    #                     background-color: #f8f9ab; 
    #                     padding: 20px 0;
    #                     margin: 10px;
    #                     margin-top: 0;
    #                     box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_goals']} - {df_filtered.iloc[i]['away_goals']}
    #                 </div>""", unsafe_allow_html=True)
    #     with col4:
    #         st.markdown(f"""
    #                 <div style="text-align: center; font-size: 15px;
    #                     background-color: #f8f9ab; 
    #                     padding: 20px 0;
    #                     margin: 10px;
    #                     margin-top: 0;
    #                     box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['away_team']}
    #                 </div>""", unsafe_allow_html=True)
    #     with col5:
    #         st.markdown(f"""
    #                 <div style="text-align: center; font-size: 15px;
    #                     background-color: #f8f9ab; 
    #                     padding: 20px 0;
    #                     margin: 10px;
    #                     margin-top: 0;
    #                     box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">Kolejka {df_filtered.iloc[i]['round']} 
    #                 </div>""", unsafe_allow_html=True)
    #     with col6:
    #         st.markdown('<div class="custom-button">', unsafe_allow_html=True)
    #         st.button(
    #             "Pokaż statystyki",
    #             key=f"button_{i}",
    #             on_click=showStats,
    #             args=(i,),
    #         )
    #         st.markdown('</div>', unsafe_allow_html=True)

        # Wyświetlanie dodatkowych informacji pod wierszem, jeśli jest włączone
        # if st.session_state.get(f"PLshow_row_{i}", False):
        #     row = df_filtered.iloc[i]
        #     st.markdown(f"""
        #             <div style="text-align: center; font-size: 15px;
        #                 background-color: #f8f9fa; 
        #                 border-radius: 10px; 
        #                 padding: 20px;
        #                 margin: 20px;
        #                 box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
        #                 <p style='text-align: center; font-size: 20px;'>Statystyki meczu {df_filtered.iloc[i]['home_team']} - {df_filtered.iloc[i]['away_team']}:</p>
        #                 <p style='text-align: center; font-size: 20px;'>Sędzia: {df_filtered.iloc[i]['referee']}</p>
        #             </div>""", unsafe_allow_html=True)
        #     categories = ["Posiadanie piłki", "Strzały", "Strzały na bramkę", "Rzuty wolne", "Rzuty rózne",
        #         "Spalone", "Faule", "Żółte kartki", "Czerwone kartki", "Podania", "Celne podania"]
        #     # trzeba będzie dodać Ball Possession jako pierwsze
        #     home_stats = ["54", df_filtered.iloc[i]["home_shots"],
        #         df_filtered.iloc[i]["home_shots_on_target"], df_filtered.iloc[i]["home_fouled"],
        #         df_filtered.iloc[i]["home_corner_kicks"], df_filtered.iloc[i]["home_offsides"], df_filtered.iloc[i]["home_fouls"],
        #         df_filtered.iloc[i]["home_cards_yellow"], df_filtered.iloc[i]["home_cards_red"],
        #         df_filtered.iloc[i]["home_passes"], df_filtered.iloc[i]["home_passes_completed"]]
        #     away_stats = ["46", df_filtered.iloc[i]["away_shots"],
        #         df_filtered.iloc[i]["away_shots_on_target"], df_filtered.iloc[i]["away_fouled"],
        #         df_filtered.iloc[i]["away_corner_kicks"], df_filtered.iloc[i]["away_offsides"], df_filtered.iloc[i]["away_fouls"],
        #         df_filtered.iloc[i]["away_cards_yellow"], df_filtered.iloc[i]["away_cards_red"],
        #         df_filtered.iloc[i]["away_passes"], df_filtered.iloc[i]["away_passes_completed"]]

        #     home_stats = [int(v) for v in home_stats]
        #     away_stats = [int(v) for v in away_stats]

        #     # Funkcja do rysowania pojedynczego wykresu dla każdej statystyki
        #     col1, col2, col3 = st.columns([2,5,2])
        #     with col2:
        #         statsGraph(home_stats, away_stats, categories)

        #     if st.button(
        #         "Pokaż więcej statystyk",
        #         key=f"PLshow_stats_button_{i}",
        #         args=(i,),
        #     ):
        #         restartStats()
        #         st.session_state["PLstats_id"] = df_filtered.iloc[i]
        #         st.switch_page("pagesHid/Statystyki Premier League.py")

    # st.write(st.session_state)