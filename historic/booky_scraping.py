import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import time
import random
from playwright.async_api import async_playwright
import asyncio

from playwright.async_api import async_playwright

async def scrape_matches(url):
    data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')
        events = soup.select(".eventListPeriodItemPartial")
        pattern = r"(\d+), '([^']+)'"
        urls = []
        for event in events:
            str = event.get("onclick")
            matches = re.search(pattern, str)
            if matches:
                event_id = matches.group(1)
                event_name = matches.group(2)
                if not event_name.lower().startswith("dzien"):
                    urls.append("https://www.etoto.pl/zaklady-bukmacherskie/pilka-nozna/anglia/premier-league/"+event_name+"/"+event_id)
        for link in urls[:2]:
            await page.goto(link)
            await page.wait_for_load_state("networkidle")
            html = await page.content()

            match = {}
            soup = BeautifulSoup(html, 'html.parser')
            # print(soup)
            correct_score_tab = soup.select_one('[game-name-for-search="dokładny wynik"]')
            
            correct_scores = correct_score_tab.select_one(".game-outcomes")
            correct_score_btn = correct_scores.select('.btn-odd')
            for score in correct_score_btn:
                match[score.select_one(".outcome-name").text] = score.select_one(".outcome-odd").text
            
            hda_tab = soup.select_one('[game-name-for-search="1x2"]')
            hda_scores = hda_tab.select_one(".game-outcomes")
            hda_btn = hda_scores.select('.btn-odd')
            match["1"] = hda_btn[0].select_one(".outcome-odd").text
            match["x"] = hda_btn[1].select_one(".outcome-odd").text
            match["2"] = hda_btn[2].select_one(".outcome-odd").text
            
            for i in range (4):
                over_tab = soup.select_one(f'[game-name-for-search="suma {i}.5 goli"]')
                over_scores = over_tab.select_one(".game-outcomes")
                over_btn = over_scores.select('.btn-odd')
                match[f"over{i}5"] = over_btn[1].select_one(".outcome-odd").text
                match[f"under{i}5"] = over_btn[0].select_one(".outcome-odd").text
                
            match["home_team"] = soup.select(".participants__item--team")[0].text
            match["away_team"] = soup.select(".participants__item--team")[1].text
            match["event_time"] = soup.select_one(".event-time").text.strip()
            data.append(match)
        await browser.close()
    return data

async def func():
    print("started.")
    data = await scrape_matches('https://www.etoto.pl/zaklady-bukmacherskie/pilka-nozna/anglia/premier-league/206')
    print("done.")
    return data

if __name__ == '__main__':
    import asyncio
    data = asyncio.run(func())
    df = pd.DataFrame(data)
    print(df)