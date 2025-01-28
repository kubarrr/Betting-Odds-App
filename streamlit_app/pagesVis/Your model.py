import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

css = """
    .stMainBlockContainer {
        padding-top: 30px !important;
    }

"""
st.html(f"<style>{css}</style>")

# Model theory

st.title("Własny model Poissona dla przewidywania wyników meczów")

st.header("Rozkład Poissona")

st.subheader("Opis modelu")
st.write(
    """Rozkład Poissona jest dyskretnym rozkładem prawdopodobieństwa, który wyraża prawdopodobieństwo uzyskania określonej liczby zdarzeń w danym przedziale czasowym, 
    pod warunkiem, że te zdarzenia zachodzą niezależnie i w stałym tempie. Prawdopodobieństwo wystąpienia \(k\) zdarzeń oblicza się wzorem:
    """
)
st.latex(r'''P(k) = \frac{e^{-\lambda} \lambda^k}{k!}''')

st.write(
    """Liczba bramek strzelanych przez drużynę w meczu, traktowanych jako niezależne i występujące z określoną średnią intensywnością, 
    może być modelowana za pomocą tego rozkładu. Dla meczu między drużyną gospodarzy H a drużyną gości A, 
    prawdopodobieństwo konkretnego wyniku jest iloczynem prawdopodobieństw dla obu drużyn:
    """
)
st.latex(r'''P(k, h) = \frac{\lambda_{\text{H}}^k e^{-\lambda_{\text{H}}}}{k!} \times \frac{\lambda_{\text{A}}^h e^{-\lambda_{\text{A}}}}{h!}''')

st.markdown("""
<h3>Przykładowa tabela rozkładu wyników meczów</h3>
<div style="display: flex; justify-content: center;">
    <table>
        <thead>
            <tr>
                <th>H/A</th>
                <td>0</td>
                <td>1</td>
                <td>2</td>
                <td>3+</td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>0.05</td>
                <td>0.08</td>
                <td>0.12</td>
                <td>0.0</td>
            </tr>
            <tr>
                <td>1</td>
                <td>0.1</td>
                <td>0.1</td>
                <td>0.1</td>
                <td>0.1</td>
            </tr>
            <tr>
                <td>2</td>
                <td>0.05</td>
                <td>0.1</td>
                <td>0.05</td>
                <td>0.05</td>
            </tr>
            <tr>
                <td>3+</td>
                <td>0.01</td>
                <td>0.02</td>
                <td>0.03</td>
                <td>0.02</td>
            </tr>
        </tbody>
    </table>
</div>   
""", unsafe_allow_html=True)

st.write("Zwycięstwo drużyny H możemy obliczyć jako sumę liczb prawdopodobieństw pod diagonalą.")


st.subheader("Wzory na obliczanie lambd")
st.write("Parametrami naszego modelu są liczby $$\lambda_{H}$$ i $$\lambda_{A}$$. Wyliczamy je przy użyciu następujących wzorów:")

st.latex(r'''\lambda_{\text{H}} = \text{SO}_{\text{H}} \cdot \text{SD}_{\text{A}} \cdot \gamma_{\text{H}}''')
st.latex(r'''\lambda_{\text{A}} = \text{SO}_{\text{A}} \cdot \text{SD}_{\text{H}} \cdot \gamma_{\text{A}}''')



st.markdown("""
    <p>Gdzie:</p>
    <ul>
        <li>λ<sub>H</sub> – oczekiwana liczba goli zdobytych przez drużynę gospodarzy,</li>
        <li>λ<sub>A</sub> – oczekiwana liczba goli zdobytych przez drużynę gości,</li>
        <li>SO<sub>H</sub> – siła ofensywna drużyny gospodarzy,</li>
        <li>SO<sub>A</sub> – siła ofensywna drużyny gości,</li>
        <li>SD<sub>H</sub> – siła defensywna drużyny gospodarzy,</li>
        <li>SD<sub>A</sub> – siła defensywna drużyny gości,</li>
        <li>γ<sub>H</sub> – średnia liczba goli strzelanych przez gospodarzy u siebie,</li>
        <li>γ<sub>A</sub> – średnia liczba goli strzelanych przez gości na wyjeździe.</li>
    </ul>
""", unsafe_allow_html=True)

# Get query params
def load_team_names_data():
    home_team = st.query_params.get("home_team", "Arsenal")
    today_date = datetime.datetime.today()
    date = st.query_params.get("date", today_date)
    matches = pd.read_csv("https://raw.githubusercontent.com/17Andri17/Betting-Odds-System/refs/heads/main/data/final_prepared_data_with_weather_new.csv")
    new_matches=pd.read_csv("https://raw.githubusercontent.com/17Andri17/Betting-Odds-System/refs/heads/main/data/new_matches_fbref.csv")
    return home_team, date, matches, new_matches

home_team, date, matches, new_matches = load_team_names_data()
date=str(pd.to_datetime(date).date())
print(date)
curr_match = matches[(matches["date"] == date) & (matches["home_team"] == home_team)]
if curr_match.empty:
    curr_match=new_matches[(new_matches["date"] == date) & (new_matches["home_team"] == home_team)]
print(curr_match["away_team"])

if not curr_match.empty:
    curr_match=curr_match.iloc[0]
    away_team=curr_match["away_team"]
    home_last5_goals=curr_match["home_last5_goals"]
    away_last5_goals=curr_match["away_last5_goals"]
else:
    away_team="Chelsea"
    home_last5_goals=1.5
    away_last5_goals=1.2

st.markdown("""
<style>

.stTextInput div[data-baseweb="input"] {
    width: 300px; 
            
}

.stNumberInput div[data-baseweb="input"] {
    width: 300px;
}

.stSlider div[data-baseweb="slider"] {
    width: 300px; 
}
table.table-bordered {
    border-collapse: collapse;
    width: 50%;
    margin: 20px;
}

table.table-bordered th, table.table-bordered td {
    text-align: center;
}
         
</style>
""", unsafe_allow_html=True)

# Providing team stats
st.subheader("Drużyny:")
home_team = st.text_input("Nazwa drużyny gospodarzy", home_team)
away_team = st.text_input("Nazwa drużyny gości", away_team)

st.subheader("Statystyki drużyny gospodarzy")
col1, col2, col3=st.columns([1, 1, 1])
with col1:
    home_avg_goals = st.number_input("Średnia liczba bramek w ostatnich 5 meczach", min_value=0.0, value=home_last5_goals, step=0.1)
with col2:
    home_offensive_power = st.number_input("Siła ofensywna", min_value=0.0, value=1.2, step=0.1)
with col3:
    home_defensive_power = st.number_input("Wskaźnik słabości defensywnej", min_value=0.0, value=0.8, step=0.1)

st.subheader("Statystyki drużyny gości")
col1, col2, col3=st.columns([1, 1, 1])
with col1:
    away_avg_goals = st.number_input("Średnia liczba bramek w ostatnich 5 meczach", min_value=0.0, value=away_last5_goals, step=0.1)
with col2:  
    away_offensive_power = st.number_input("Siła ofensywna", min_value=0.0, value=1.1, step=0.1)
with col3:    
    away_defensive_power = st.number_input("Wskaźnik słabości defensywnej", min_value=0.0, value=0.9, step=0.1)


home_lambda = home_offensive_power * away_defensive_power * home_avg_goals
away_lambda = away_offensive_power * home_defensive_power * away_avg_goals


# Calculating probs
st.subheader(f"Obliczone lambdy:")
st.write(f"##### Drużyna gospodarzy {home_team}: {home_lambda:.2f}")
st.write(f"##### Drużyna gości {away_team}: {away_lambda:.2f}")


st.subheader("Macierz prawdopodobieństw:")
max_goals = st.slider("Maksymalna liczba bramek do wyświetlenia", min_value=3, max_value=7, value=5)

home_goals = np.arange(0, max_goals + 1)
away_goals = np.arange(0, max_goals + 1)

probability_matrix = np.zeros((len(home_goals), len(away_goals)))

for i, hg in enumerate(home_goals):
    for j, ag in enumerate(away_goals):
        probability_matrix[i, j] = poisson.pmf(hg, home_lambda) * poisson.pmf(ag, away_lambda)


probability_display = np.vectorize(lambda x: f"{np.round(x * 100, 2)}%")(probability_matrix)
probability_data = pd.DataFrame(probability_display, index=home_goals, columns=away_goals)

results_table = "<table class='table table-bordered'>"
results_table += "<thead><tr><th>H/A</th>" + "".join(f"<th>{col}</th>" for col in probability_data.columns) + "</tr></thead>"
results_table += "<tbody>"

for i, (row_name, row_values) in enumerate(probability_data.iterrows()):
    results_table += f"<tr><th>{row_name}</th>"
    for j, value in enumerate(row_values):
        if i == j:
            cell_style = "background-color: #b9bbbc;"
        elif i > j:
            cell_style = "background-color: #6dff90;"
        else:
            cell_style = "background-color: #72c3fb;"
        results_table += f"<td style='{cell_style}'>{value}</td>"
    results_table += "</tr>"

results_table += "</tbody></table>"

st.markdown(f"""
    <h3 style='text-align: center;'>{home_team} vs {away_team}</h3>
    <div style="display: flex; justify-content: center;">
        {results_table}
    </div>
""", unsafe_allow_html=True)

home_win_prob = int(np.sum(np.tril(probability_matrix, -1))*100)
draw_prob = int(np.sum(np.diag(probability_matrix))*100)
away_win_prob = 100-home_win_prob-draw_prob


probabilities=[home_win_prob, draw_prob, away_win_prob]
colors = ['#6dff90', '#b9bbbc', '#72c3fb']
fig1, ax = plt.subplots(figsize=(4, 1))
start = 0

for prob, color in zip(probabilities, colors):
    ax.barh(0, prob/100, left=start, color=color, edgecolor='none', height=0.5)
    start += prob/100

start = 0
for prob, color in zip(probabilities, colors):
    ax.text(start + prob / 200, 0, f"{prob}%", color='black', va='center', ha='center', fontsize=10)
    start += prob/100

ax.set_xlim(0, 1)
ax.axis('off') 
plt.title('Prawdopodbieństwa:', pad=10, fontsize=8)
plt.show()
plt.tight_layout()
col1, col2, col3=st.columns([1, 2, 1])
with col2:
    st.pyplot(fig1)



