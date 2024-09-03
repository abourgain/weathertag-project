"""File to load weather data and lookup weather for a city"""

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import pycountry
import streamlit as st


def get_country_name(country_code):
    """Convert a country code to the full country name."""
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name.replace(",", "") if country else "Unknown country code"
    except KeyError:
        return "Unknown country code"


@st.cache_data
def load_weather_data(save_local: bool = False):
    """Load and index weather data from the gzip file, and cache the result."""
    weather_data_cache, locations_data_cache, coord_data_cache = None, None, None

    # Load cached data from files if save_local is True and files exist
    if save_local and Path("./data/locations.json").exists():
        with open("./data/locations.json", "r", encoding="utf-8") as f:
            locations_data_cache = json.load(f)

    if save_local and Path("./data/coords.json").exists():
        with open("./data/coords.json", "r", encoding="utf-8") as f:
            coord_data_cache_str_keys = json.load(f)
        coord_data_cache = {tuple(map(float, key.strip("()").split(", "))): value for key, value in coord_data_cache_str_keys.items()}

    if save_local and Path("./data/weather.json").exists():
        with open("./data/weather.json", "r", encoding="utf-8") as f:
            weather_data_cache_str_keys = json.load(f)
        weather_data_cache = {tuple(map(str, key.strip("()").replace("'", "").split(", "))): value for key, value in weather_data_cache_str_keys.items()}

    # Return cached data if loaded from files
    if weather_data_cache and locations_data_cache and coord_data_cache:
        return weather_data_cache, locations_data_cache, coord_data_cache

    # If not loaded from files, compute the data
    weather_data_cache, locations_data_cache, coord_data_cache = {}, {}, {}

    with gzip.open(Path("./data/daily_14.json.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            city, country_code, coord = row["city"]["name"].replace(",", ""), row["city"]["country"], row["city"]["coord"]
            city_key = (city, country_code)
            data = row["data"]
            data_dict = {str(datetime.fromtimestamp(d["dt"], timezone.utc).date()): d for d in data}
            weather_data_cache[city_key] = {
                "city": row["city"],
                "time": row["time"],
                "data": data_dict,
            }

            if country_code not in locations_data_cache:
                locations_data_cache[country_code] = set()
            locations_data_cache[country_code].add(city)

            coord = (coord["lat"], coord["lon"])
            if coord not in coord_data_cache:
                coord_data_cache[coord] = (city, country_code)

    for country_code in locations_data_cache:
        locations_data_cache[country_code] = sorted(locations_data_cache[country_code])

    # Optionally save computed data to files if save_local is True
    if save_local:
        if not Path("./data/locations.json").exists():
            with open("./data/locations.json", "w", encoding="utf-8") as f:
                json.dump(locations_data_cache, f)

        if not Path("./data/coords.json").exists():
            with open("./data/coords.json", "w", encoding="utf-8") as f:
                coord_data_cache_str_keys = {str(key): value for key, value in coord_data_cache.items()}
                json.dump(coord_data_cache_str_keys, f)

        if not Path("./data/weather.json").exists():
            with open("./data/weather.json", "w", encoding="utf-8") as f:
                weather_data_cache_str_keys = {str(key): value for key, value in weather_data_cache.items()}
                json.dump(weather_data_cache_str_keys, f)

    return weather_data_cache, locations_data_cache, coord_data_cache


def lookup_weather(dt, weather_data_cache, city_name: str = "London", country_code: str = "GB"):
    """Lookup the weather for a specific city and date."""

    city_key = (city_name, country_code)
    if city_key not in weather_data_cache:
        return "City not found in the weather data..."

    if str(dt) not in weather_data_cache[city_key]["data"]:
        return "Weather data not found for the selected date..."

    weather = {
        "city": {"name": city_name, "country": get_country_name(country_code)},
        "weather": weather_data_cache[city_key]["data"][str(dt)]["weather"][0]["main"],
        "temp_day": f'{weather_data_cache[city_key]["data"][str(dt)]["temp"]["day"] - 273.15:.2f}C',
    }
    return weather
