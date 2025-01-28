# Betting-Odds-System

<h4>Dane które zbieramy:</h4>
<ul>
  <li>Statystyki meczowe (zrobione dla PL 17-24 ze stronki FBRef)</li>
  <li>Dane stadionów (dodać wymiary boisk niektóych stadionów)</li>
  <li>Dane pogodowe (zrobione, będzie można aktualizować dane w czasie rzeczywistym)</li>
  <li>Dane zawodników (do zrobienia)</li>
  <li>Dane tekstowe do analizy sentymentu (do zastanowienia)</li>
</ul>


<h4>Wstępny pomysł na sieć (Model 1 V1):</h4>
Input 1: Statystyki meczowe z poprzednich 5-10 meczów</br>
Input 2: Agregacja statystyk meczowych z ostatnich 30-50 meczów</br>
Input 3: Motywacja/zmęczenie drużyny: ilość dni od ostatniego meczu, waga meczu, ilość dni do kolejnego kluczowego meczu, itp.</br>
Input 4: Dane pogodowe/stadionowe z odpowiednim embeddingiem</br>
Input 5: Sentyment z mediów, opcjonalnie przepuszczony przez LTSM</br>
Input 6: Dane dotyczące zawodników -> Dodatkowy model do tego lub jakaś miara wpływu nieobecnych zawodników względem obecnych</br>
</br>
- 6 inputów przepuszczamy przez Dense layers (niektóre ewentualnie przez LTSM, ale nie wiem jeszcze jak to dokładnie działa xd)</br>
- Łączymy dane za pomocą Concatenate</br>
- Kilka dense layers z coraz mniejszą ilością neuronów</br>
- Na koniec Softmax layer z 3 wyjściami odpowiadającymi prawdopodobieństwu wygranej gospodarzy/remisu/wygranej gości</br>

<h4>Nasze własne metryki:</h4>
<ul>
  <li>Dni do najbliższego ważnego meczu</li>
  <li>Dni od ostatniego meczu (dla zawodników/drużyny)</li>
  <li>Jak sobie radzi drużyna w zależności od pogody</li>
  <li>Zależność stylu gry</li>
  <li>Statystyczne porównanie formacji</li>
  <li>Wpływ sędziego - więcej kartek: trichę większa losowość wyniku</li>
</ul>
