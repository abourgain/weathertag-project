"""Main WeatherTag app."""

#!/usr/bin/env -S conda run -n weathertag_env --live-stream streamlit run
import datetime
import os

import streamlit as st  # See: https://docs.streamlit.io/library/api-reference
from dotenv import load_dotenv

from components import wikipedia_current_events_page
from components.events_extractor import (
    computer_event_duplication_rate,
    extract_current_events,
    n_events_old_extractor,
)
from components.llm import LocationExtractionManager, TitleExtractorManager
from components.location_embeddings import LocationEmbeddingManager
from components.weather import get_country_name, load_weather_data, lookup_weather


load_dotenv()


@st.cache_data
def load_weather_data_cached():
    """Load weather data and locations data, cached to avoid repeated loading."""
    return load_weather_data()


def init_managers():
    """Initialize manager classes and store them in session state."""
    if "location_extraction_manager" not in st.session_state:
        st.session_state.location_extraction_manager = LocationExtractionManager() if "OPENAI_API_KEY" in os.environ else None
    if "location_embedding_manager" not in st.session_state:
        st.session_state.location_embedding_manager = LocationEmbeddingManager()
    if "title_extraction_manager" not in st.session_state:
        st.session_state.title_extraction_manager = TitleExtractorManager() if "OPENAI_API_KEY" in os.environ else None


def weathertag_app():  # pylint: disable=too-many-locals
    """Main WeatherTag app."""
    st.set_page_config(layout="wide")

    # Load cached data
    weather_data_cache, locations_data_cache, coord_data_cache = load_weather_data_cached()

    # Initialize managers and store them in session state
    init_managers()

    """
    # WeatherTag app

    [Wikipedia Current events](https://en.wikipedia.org/wiki/Portal:Current_events) pages enumerate noteworthy worldwide events for every single day [since 2002](https://en.wikipedia.org/wiki/Portal:Current_events/January_2002).
    An event entry in those pages typically describes what happened where and when.
    Suppose, we need to see the weather information alongside every event for some very important reason.
    Given a geolocation and date/time, it's fairly easy to get the historic weather data from [OpenWeather](https://openweathermap.org) through their API as well as [bulk data access](https://openweathermap.org/bulk).

    In this exercise, let's build an app that can tag events with their weather info.
    In concrete terms, when clicking on the `show_current_events_tagged_with_weather()` option below,
    we would like to see a clean list of events, and next to each event what the weather was like.
    You can evolve this skeleton code which already knows how to fetch and render the Wikipedia Current events page for any given date.
    This skeleton streamlit app should be self-explanatory but if you'd like to learn more, see the [Streamlit API docs](https://docs.streamlit.io/library/api-reference).

    ----
    """

    if "OPENAI_API_KEY" in os.environ:
        # Select the method
        ai = st.toggle("Activate extraction with AI")
    else:
        ai = False

    location_manager = st.session_state.location_extraction_manager if ai else st.session_state.location_embedding_manager
    title_manager = st.session_state.title_extraction_manager if ai else None

    if ai:
        st.write("AI extraction is activated. Using advanced language models for location and title extraction.")
    else:
        st.write("AI extraction is deactivated. Using traditional embedding-based location extraction.")

    fns = [
        show_actual_page,
        show_current_events_extracted,
        show_current_events_page_and_extracted_side_by_side,
        show_weather_for_a_place,
        show_current_events_tagged_with_weather,
    ]
    i = st.radio(
        "Select the function in `weather.py` to run:",
        options=range(len(fns)),
        format_func=lambda i: f"{fns[i].__name__}() -- {(fns[i].__doc__ or '').strip()}",
    )
    fn = fns[i]

    # date picker
    cols = st.columns((4, 1, 1))
    with cols[0]:
        dt = st.date_input(
            "Select the date to see:",
            value=datetime.datetime(2017, 3, 22),
            # NOTE http://bulk.openweathermap.org/sample/ daily_14.json.gz is roughly from 2017/03/13 to 03/29
            key="dt",
        )
    with cols[1]:

        def prev_dt():
            st.session_state.dt -= datetime.timedelta(days=1)  # pylint: disable=no-member

        st.button("Previous Day", on_click=prev_dt)

    with cols[2]:

        def next_dt():
            st.session_state.dt += datetime.timedelta(days=1)  # pylint: disable=no-member

        st.button("Next Day", on_click=next_dt)

    # grab the chosen date's current events page (just the actual content; wikitext parsed in html)
    html, wp_url = wikipedia_current_events_page(dt)

    """
    ----
    """

    fn(**vars())  # NOTE this passes down to the selected function all variables in scope (vars())


def show_actual_page(html, wp_url, **kwargs):
    """
    Render original Wikipedia page.
    """
    st.markdown(f"""Wikipedia page: <{wp_url}>""")
    html_quoted = f"""
        <base href="{wp_url}"/>
        <style>.mw-parser-output {{ color: black; background-color: white; }}</style>
        {html}
        """.replace(
        '"', "&quot;"
    ).replace(
        "\n", "&#10;"
    )
    st.markdown(
        f"""
        <iframe srcdoc="{html_quoted}" src="{wp_url}" style="width:100%; min-height:800px; border: 5px inset;"></iframe>
        """,
        unsafe_allow_html=True,
    )


def show_current_events_extracted(**kwargs):
    """
    Extract and show a list of events.
    """
    with st.spinner("Extracting events details..."):
        events = list(extract_current_events(**kwargs))
    n_events_old = n_events_old_extractor(**kwargs)
    duplication_rate = computer_event_duplication_rate(n_events_old, len(events))
    st.write("Previous method:")
    st.write(f"Event duplication rate: {duplication_rate:.2%} ({n_events_old} events extracted with the previous method, compared to {len(events)} now)")
    """
    ----
    """
    st.write("Current method:")
    st.write(events)


def show_current_events_page_and_extracted_side_by_side(**kwargs):
    """
    Render the original Wikipedia page and the extractions side by side.
    """
    cols = st.columns(2)
    with cols[0]:
        show_actual_page(**kwargs)
    with cols[1]:
        show_current_events_extracted(**kwargs)


def show_weather_for_a_place(dt, weather_data_cache, locations_data_cache, **kwargs):
    """
    Lookup weather data for a place.
    """
    # with st.spinner("Loading weather data..."):
    #     _, locations_data_cache = load_weather_data()

    # Create a dictionary that maps country names to country codes
    country_name_to_code = {get_country_name(code): code for code in locations_data_cache.keys()}

    # Streamlit selectbox with sorted country names
    country_name = st.selectbox("Select a country", sorted(country_name_to_code.keys()))
    country_code = country_name_to_code[country_name]

    # Select a city based on the selected country
    city_name = st.selectbox("Select a city", locations_data_cache[country_code])

    weather = lookup_weather(dt, weather_data_cache, city_name, country_code)
    st.write(weather)


def show_current_events_tagged_with_weather(**kwargs):
    """
    Extract and show a list of events along with the historic weather info.
    """
    with st.spinner("Extracting events details and weather info..."):
        events = list(extract_current_events(**kwargs))

    for event in events:
        cols = st.columns(2)
        with cols[0]:
            st.write(event)
        with cols[1]:
            weathers = []
            if "locations" in event:
                for location in event["locations"]:
                    weather = lookup_weather(
                        kwargs["dt"],
                        kwargs["weather_data_cache"],
                        location.city,
                        location.country,
                    )
                    weathers.append(weather)

            st.write(weathers)


if __name__ == "__main__":
    weathertag_app()
