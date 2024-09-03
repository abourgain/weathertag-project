"""Import all components here to make them available in the main app file."""

import requests

import streamlit as st


@st.cache_data
def wikipedia_current_events_page(dt):
    wp_dt = dt.strftime("%Y_%B_") + dt.strftime("%e").strip()  # XXX only `2024_April_1` works not `2024_April_01`
    wp_title = f"Portal:Current_events/{wp_dt}"
    wp_url = f"https://en.wikipedia.org/wiki/{wp_title}"
    # html = requests.get(wp_url).text  # XXX this includes too much boilerplate html around the content
    wp_api_url = f"https://en.wikipedia.org/w/api.php?action=parse&format=json&page={wp_title}"
    res = requests.get(wp_api_url, timeout=10)
    html = res.json()["parse"]["text"]["*"]
    return html, wp_url
