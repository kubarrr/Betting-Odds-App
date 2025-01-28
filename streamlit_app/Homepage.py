import streamlit as st
import numpy as np
import pandas as pd
from models import FootballMatchPredictor, FootballMatchPredictorOutcome

st.set_page_config(layout="wide")

homepage = st.Page(
    "Kursomat.py",
    title="Strona Główna",
    icon="🏠",
    default=True,
)

premier_league = st.Page(
    "pagesVis/Premier League.py",
    title="Premier League",
    icon="⚽",
)

bundesliga = st.Page(
    "pagesVis/Bundesliga.py",
    title="Bundesliga",
    icon="⚽",
)

seriea = st.Page(
    "pagesVis/Serie A.py",
    title="Serie A",
    icon="⚽",
)

ligue1 = st.Page(
    "pagesVis/Ligue 1.py",
    title="Ligue 1",
    icon="⚽",
)

laliga = st.Page(
    "pagesVis/La Liga.py",
    title="La Liga",
    icon="⚽",
)

historic_match_stats = st.Page(
    "pagesHid/Statystyki Przedmeczowe.py",
    title="Statystyki Przedmeczowe",
    icon="📊",
)

future_match_stats = st.Page(
    "pagesHid/Statystyki Pomeczowe.py",
    title="Statystyki Pomeczowe",
    icon="📊",
)

model_page = st.Page(
    "pagesVis/Your model.py",
    title="Your model",
    icon="📊",
)

pg = st.navigation(pages = [homepage, premier_league, bundesliga, seriea, ligue1, laliga, historic_match_stats, future_match_stats, model_page], position="hidden")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 200px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Wybierz stronę:")

# if st.sidebar.button(
#             "Strona Główna",
#             key=f"Home"
#         ):
#             st.switch_page("Kursomat.py")

# if st.sidebar.button(
#             "Bundesliga",
#             key=f"Bundesliga"
#         ):
#             st.switch_page("pagesVis/Bundesliga.py")

# if st.sidebar.button(
#             "La Liga",
#             key=f"La Liga"
#         ):
#             st.switch_page("pagesVis/La Liga.py")

# if st.sidebar.button(
#             "Ligue 1",
#             key=f"Ligue1"
#         ):
#             st.switch_page("pagesVis/Ligue 1.py")

# if st.sidebar.button(
#             "Premier League",
#             key=f"PremierLeague"
#         ):
#             st.switch_page("pagesVis/Premier League.py")

# if st.sidebar.button(
#             "Serie A",
#             key=f"SerieA"
#         ):
#             st.switch_page("pagesVis/Serie A.py")

# if st.sidebar.button(
#             "Stwórz własny model",
#             key=f"OwnModel"
#         ):
#             st.switch_page("pagesVis/Stwórz własny model.py")

with st.sidebar:
    st.page_link("Kursomat.py", label="Strona główna", icon="🏠")
    st.page_link("pagesVis/Bundesliga.py", label="Bundesliga", icon="⚽")
    st.page_link("pagesVis/La Liga.py", label="La Liga", icon="⚽")
    st.page_link("pagesVis/Ligue 1.py", label="Ligue 1", icon="⚽")
    st.page_link("pagesVis/Premier League.py", label="Premier League", icon="⚽")
    st.page_link("pagesVis/Serie A.py", label="Serie A", icon="⚽")
    st.page_link("pagesVis/Your model.py", label="Stwórz własny model", icon="📊")

pg.run()