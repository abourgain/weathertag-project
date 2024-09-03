"""Content for the WeatherTag app."""

INTRO_TEXT = """
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

SEPARATOR = """----"""
