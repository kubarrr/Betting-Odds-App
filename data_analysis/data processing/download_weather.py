#Data scrapping from https://open-meteo.com/
import requests

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/era5"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(latitude, longitude, start_hour, end_hour, params, type_url='historical'):
#latitude=52.52,48.85&longitude=13.41,2.35,
#start_hour="2022-06-30T12:00"
#params &hourly=['temperature_2m', 'precipitation', 'wind_speed_10m', 'relative_humidity_2m', 'cloud_cover']
#type_url=historical or forecast
    params = ','.join(params)
    filter = f'latitude={latitude}&longitude={longitude}&hourly={params}&start_hour={start_hour}&end_hour={end_hour}'
    if type_url=='forecast':
        download_url=FORECAST_URL
    else:
        #We assume that generally we want to download historical data
        download_url=HISTORICAL_URL
    req=requests.get(download_url+'?'+filter)
    if req.status_code == 200:
        data=req.json()
        return {**data['hourly']}
    else:
        print(f"Błąd: {req.status_code}")
        return None


