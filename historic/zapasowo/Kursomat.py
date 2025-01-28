import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

@st.cache_data
def loadData():
    df = pd.read_csv("../prepared_data.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Najpierw konwersja do datetime
    df["date"] = df["date"].astype(str)
    df["formation_home"] = df["formation_home"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace(r"-1-1$", "-2", regex=True)
    df["formation_home"] = df["formation_home"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["formation_away"] = df["formation_away"].str.replace("4-1-2-1-2", "4-3-1-2", regex=True)
    df["round"] = df["round"].astype(int)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    dfPL = df[df["league"] == "pl"]
    dfLL = df[df["league"] == "ll"]
    dfL1 = df[df["league"] == "l1"]
    dfBun = df[df["league"] == "bl"]
    dfSA = df[df["league"] == "sa"]

    players_23_24 = pd.read_csv("../fbref/data/players_pl_23-24_fbref.csv")
    players_22_23 = pd.read_csv("../fbref/data/players_pl_22-23_fbref.csv")
    players_21_22 = pd.read_csv("../fbref/data/players_pl_21-22_fbref.csv")
    players_20_21 = pd.read_csv("../fbref/data/players_pl_20-21_fbref.csv")
    players_19_20 = pd.read_csv("../fbref/data/players_pl_19-20_fbref.csv")
    players_18_19 = pd.read_csv("../fbref/data/players_pl_18-19_fbref.csv")
    players = pd.concat([players_23_24, players_22_23, players_21_22, players_20_21, players_19_20, players_18_19], ignore_index=True)
    players = players.rename(columns={"position": "position_x"})
    players["date"] = pd.to_datetime(players["date"], errors="coerce")  # Najpierw konwersja do datetime
    players["date"] = players["date"].astype(str)

    standings = pd.read_csv("../standings.csv")
    standings['date']=pd.to_datetime(standings['date'])
    standings['goal_difference'] = standings['goal_difference'].astype(int)
    standings['goals'] = standings['goals'].astype(int) 
    standings['goals_conceded'] = standings['goals_conceded'].astype(int)
    standingsPL = standings[standings["league"] == "pl"]
    standingsLL = standings[standings["league"] == "ll"]
    standingsL1 = standings[standings["league"] == "l1"]
    standingsBun = standings[standings["league"] == "bl"]
    standingsSA = standings[standings["league"] == "sa"]

    odds = pd.read_csv("../odds.csv")
    oddsPL = odds[odds["Div"] == "E0"]
    oddsLL = odds[odds["Div"] == "SP1"]
    oddsL1 = odds[odds["Div"] == "F1"]
    oddsBun = odds[odds["Div"] == "D1"]
    oddsSA = odds[odds["Div"] == "I1"]

    return dfPL, dfLL, dfL1, dfBun, dfSA, standingsPL, standingsLL, standingsL1, standingsBun, standingsSA, players, oddsPL, oddsLL, oddsL1, oddsBun, oddsSA

# Sprawdzenie, czy dane są już w session_state
if "dfPL" not in st.session_state:
    dfPL, dfLL, dfL1, dfBun, dfSA, standingsPL, standingsLL, standingsL1, standingsBun, standingsSA, players, oddsPL, oddsLL, oddsL1, oddsBun, oddsSA = loadData()
    st.session_state["dfPL"] = dfPL
    st.session_state["standingsPL"] = standingsPL
    st.session_state["playersPL"] = players
    st.session_state["oddsPL"] = oddsPL

    st.session_state["dfLL"] = dfLL
    st.session_state["standingsLL"] = standingsLL
    #st.session_state["playersPL"] = players
    st.session_state["oddsLL"] = oddsPL

    st.session_state["dfL1"] = dfL1
    st.session_state["standingsL1"] = standingsL1
    #st.session_state["playersPL"] = players
    st.session_state["oddsL1"] = oddsL1

    st.session_state["dfBun"] = dfBun
    st.session_state["standingsBun"] = standingsBun
    #st.session_state["playersPL"] = players
    st.session_state["oddsBun"] = oddsBun

    st.session_state["dfSA"] = dfSA
    st.session_state["standingsSA"] = standingsSA
    #st.session_state["playersPL"] = players
    st.session_state["oddsSA"] = oddsSA

st.title("Witaj użytkowniku!")
st.write("""Witaj na stronie, na której możesz sprawdzić kursy na mecze
    piłkarskie z różnych lig europejskich, a także przeprowadzić analizę spotkań różnych drużyn.""")

st.subheader("Wybierz ligę, dla której chcesz zobaczyć statystyki:")


# Define local image paths and text content
columns_data = [
    {"image_path": "graphics/Germany.png", "text": "Bundesliga", "button_label": "Przejdź do strony"},
    {"image_path": "graphics/Spain.png", "text": "La Liga", "button_label": "Przejdź do strony"},
    {"image_path": "graphics/France.png", "text": "Ligue 1", "button_label": "Przejdź do strony"},
    {"image_path": "graphics/England.png", "text": "Premier League", "button_label": "Przejdź do strony"},
    {"image_path": "graphics/Italy.png", "text": "Serie A", "button_label": "Przejdź do strony"}
]

# Create 5 columns
cols = st.columns(5)

# Populate each column with content
for i, col in enumerate(cols):
    with col:
        try:
            image = Image.open(columns_data[i]['image_path'])
            st.image(image.resize((1280,854)), use_container_width=True)

        except FileNotFoundError:
            col.error(f"Nie znaleziono obrazu: {columns_data[i]['image_path']}")

cols = st.columns(5)
for i, col in enumerate(cols):
    with col:
        if st.button(
                f"{columns_data[i]['text']}",
                key=f"HomeButton{i}"
            ):
                st.switch_page(f"pagesVis/{columns_data[i]['text']}.py")

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