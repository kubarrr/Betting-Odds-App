import streamlit as st
import numpy as np
import pandas as pd
import runpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if st.session_state.get("dfPL", None) is None:
    st.write("Odwiedź najpierw stronę główną aby załadować dane")
else:
    # Chowanie statystyk po zmianie filtrów
    def restartStats():
        for i in range (st.session_state["PLnumber_of_matches"]):
            if f"PLshow_row_{i}" in st.session_state:
                st.session_state[f"PLshow_row_{i}"] = False

    # Pokazywanie statystyk dla i-tego meczu
    def showStats(i):
        st.session_state[f"PLshow_row_{i}"] = not st.session_state.get(f"PLshow_row_{i}", False)

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

    df = st.session_state["dfPL"].copy()
    standings = st.session_state["standingsPL"].copy()
    df_filtered=df.copy()
    standings_filtered=standings.copy()

    if "PLseason_filter" not in st.session_state:
        st.session_state["PLseason_filter"] = []
    if "PLteam_filter" not in st.session_state:
        st.session_state["PLteam_filter"] = []
    if "PLnumber_of_matches" not in st.session_state:
        st.session_state["PLnumber_of_matches"] = 10


    # Tytuł i tworzenie filtrów
    st.title("Premier League")
    # Filtry dla tabeli
    col1, col2 = st.columns(2)
    with col1:
        season_filter = st.multiselect("Wybierz sezon, z którego chcesz zobaczyć tabelę oraz statystyki",
            options = standings['season'].unique(), on_change=showDateButton, max_selections=1, default=st.session_state["PLseason_filter"], key="PLseason_filter")

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
    st.subheader(f"Tabela Premier League w sezonie {season_filter_matches}")
    st.caption(f"Stan na: {date_standings}")
    date_standings = pd.to_datetime(date_standings)

    selected_columns_standings = ['team', 'matches_played', 'wins', 'draws', 'defeats', 'goal_difference', 'goals', 'goals_conceded', 'points']
    table = standings_filtered[selected_columns_standings]
    table['place'] = range(1, len(table) + 1)
    table = table.set_index('place')
    table.columns = [ 'Zespół', 'Mecze rozegrane', 'Wygrane', 'Remisy', 'Porażki', 'Różnica bramek', 'Bramki strzelone', 'Bramki stracone', 'Punkty']
    col1, col2, col3 = st.columns([1,5,1])
    with col2:
        st.table(table)

    # Filtry dla meczów
    filtr1, filtr2 = st.columns(2)

    with filtr1:
        team_filter = st.multiselect("Wybierz drużynę", options = sorted(df_filtered['home_team'].unique()), default=st.session_state["PLteam_filter"], key="PLteam_filter")
    with filtr2:
        number_of_matches = st.slider("Wybierz liczbę wyświetlanych meczów", min_value=10, max_value=100, step=5, value=st.session_state["PLnumber_of_matches"], key="PLnumber_of_matches")

    team_filter2=team_filter


    # Filtrowanie danych

    if team_filter==[]:
        team_filter2=df['home_team'].unique()
    df_filtered=df[(pd.to_datetime(df['date'])<=date_standings)
                & (df['season'] == season_filter_matches)
                & ((df['home_team'].isin(team_filter2))
                    | (df['away_team'].isin(team_filter2)))]
    df_filtered.sort_values(by=["date", "time", "round"], ascending=False, inplace=True)

    # Wypisywanie danych
    for i in range(min(number_of_matches, df_filtered['home_team'].count())):
        col1, col2, col3, col4, col5, col6 = st.columns([3,5,2,5,2,2])
        with col1:
            st.markdown(f"""
                    <div style="text-align: center; font-size: 15px;
                        background-color: #f8f9ab; 
                        padding: 20px 0;
                        margin: 10px;
                        margin-top: 0;
                        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['date']} {df_filtered.iloc[i]['time']}
                    </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                    <div style="text-align: center; font-size: 15px;
                        background-color: #f8f9ab; 
                        padding: 20px 0;
                        margin: 10px;
                        margin-top: 0;
                        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_team']}
                    </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                    <div style="text-align: center; font-size: 15px;
                        background-color: #f8f9ab; 
                        padding: 20px 0;
                        margin: 10px;
                        margin-top: 0;
                        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['home_goals']} - {df_filtered.iloc[i]['away_goals']}
                    </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                    <div style="text-align: center; font-size: 15px;
                        background-color: #f8f9ab; 
                        padding: 20px 0;
                        margin: 10px;
                        margin-top: 0;
                        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">{df_filtered.iloc[i]['away_team']}
                    </div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
                    <div style="text-align: center; font-size: 15px;
                        background-color: #f8f9ab; 
                        padding: 20px 0;
                        margin: 10px;
                        margin-top: 0;
                        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">Kolejka {df_filtered.iloc[i]['round']} 
                    </div>""", unsafe_allow_html=True)
        with col6:
            st.markdown('<div class="custom-button">', unsafe_allow_html=True)
            st.button(
                "Pokaż statystyki",
                key=f"button_{i}",
                on_click=showStats,
                args=(i,),
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Wyświetlanie dodatkowych informacji pod wierszem, jeśli jest włączone
        if st.session_state.get(f"PLshow_row_{i}", False):
            row = df_filtered.iloc[i]
            st.markdown(f"""
                    <div style="text-align: center; font-size: 15px;
                        background-color: #f8f9fa; 
                        border-radius: 10px; 
                        padding: 20px;
                        margin: 20px;
                        box-shadow: 4px 4px 8px rgba(0.2, 0.2, 0.2, 0.2);">
                        <p style='text-align: center; font-size: 20px;'>Statystyki meczu {df_filtered.iloc[i]['home_team']} - {df_filtered.iloc[i]['away_team']}:</p>
                        <p style='text-align: center; font-size: 20px;'>Sędzia: {df_filtered.iloc[i]['referee']}</p>
                    </div>""", unsafe_allow_html=True)
            categories = ["Posiadanie piłki", "Strzały", "Strzały na bramkę", "Rzuty wolne", "Rzuty rózne",
                "Spalone", "Faule", "Żółte kartki", "Czerwone kartki", "Podania", "Celne podania"]
            # trzeba będzie dodać Ball Possession jako pierwsze
            home_stats = ["54", df_filtered.iloc[i]["home_shots"],
                df_filtered.iloc[i]["home_shots_on_target"], df_filtered.iloc[i]["home_fouled"],
                df_filtered.iloc[i]["home_corner_kicks"], df_filtered.iloc[i]["home_offsides"], df_filtered.iloc[i]["home_fouls"],
                df_filtered.iloc[i]["home_cards_yellow"], df_filtered.iloc[i]["home_cards_red"],
                df_filtered.iloc[i]["home_passes"], df_filtered.iloc[i]["home_passes_completed"]]
            away_stats = ["46", df_filtered.iloc[i]["away_shots"],
                df_filtered.iloc[i]["away_shots_on_target"], df_filtered.iloc[i]["away_fouled"],
                df_filtered.iloc[i]["away_corner_kicks"], df_filtered.iloc[i]["away_offsides"], df_filtered.iloc[i]["away_fouls"],
                df_filtered.iloc[i]["away_cards_yellow"], df_filtered.iloc[i]["away_cards_red"],
                df_filtered.iloc[i]["away_passes"], df_filtered.iloc[i]["away_passes_completed"]]

            home_stats = [int(v) for v in home_stats]
            away_stats = [int(v) for v in away_stats]

            # Funkcja do rysowania pojedynczego wykresu dla każdej statystyki
            col1, col2, col3 = st.columns([2,5,2])
            with col2:
                statsGraph(home_stats, away_stats, categories)

            if st.button(
                "Pokaż więcej statystyk",
                key=f"PLshow_stats_button_{i}",
                args=(i,),
            ):
                restartStats()
                st.session_state["PLstats_id"] = df_filtered.iloc[i]
                st.switch_page("pagesHid/Statystyki Premier League.py")

    st.write(st.session_state)