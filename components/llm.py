"""Location Extraction Manager to extract locations from event descriptions and find the closest cities."""

import json
import math
import os
from typing import List, Optional

import requests
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator


from dotenv import load_dotenv

from components.config import logging

load_dotenv()


class Location(BaseModel):
    """Location model to represent a city and country."""

    city: str = Field(description="city where the event occurred")
    country: str = Field(description="country where the event occurred, in alpha-2 code")


class Locations(BaseModel):
    """Locations model to represent a list of locations."""

    locations: List[Optional[Location]] = Field(description="list of locations where the event happened")

    @field_validator('locations')
    def not_start_with_number(cls, locations):  # pylint: disable=no-self-argument
        """Validate that the city name does not start with a number."""
        if locations and locations[0].city[0].isnumeric():
            raise ValueError("The city name cannot start with numbers!")
        return locations


class Title(BaseModel):
    title: str = Field(description="the extracted news headline")


class LocationExtractionManager:
    """Class to manage location extraction, validation, and finding closest cities."""

    def __init__(self, locations_data_cache: dict = None, coord_data_cache: dict = None):
        """Initialize the LocationExtractionManager with required components."""
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
        self.parser = self._setup_parser()
        self.chain = self._setup_chain()

        self.locations_data = self._load_locations_data() if not locations_data_cache else locations_data_cache
        self.coord_data_cache = self._load_coords_data() if not coord_data_cache else coord_data_cache

    def _load_locations_data(self):
        """Load location data from a JSON file."""
        locations_file = "./data/locations.json"

        if os.path.exists(locations_file):
            with open(locations_file, "r", encoding="utf-8") as f:
                locations_data = json.load(f)

            return locations_data
        raise FileNotFoundError("Locations data file not found!")

    def _load_coords_data(self):
        """Load coordinates data from a JSON file."""
        coords_file = "./data/coords.json"

        if os.path.exists(coords_file):
            with open(coords_file, "r", encoding="utf-8") as f:
                coord_data_cache_str_keys = json.load(f)

            # Convert the keys from strings to tuples
            coord_data_cache = {tuple(map(float, key.strip("()").split(", "))): value for key, value in coord_data_cache_str_keys.items()}

            return coord_data_cache
        raise FileNotFoundError("Coordinates data file not found!")

    def _setup_parser(self):
        """Set up the Pydantic output parser."""

        return PydanticOutputParser(pydantic_object=Locations)

    def _setup_chain(self):
        """Set up the LLM chain with prompts and examples."""
        examples = [
            {
                "event": "During the 2024 World Health Summit in Tokyo, global leaders discussed the ongoing health crisis that started in the Middle East and its effects on neighboring countries. The discussions emphasized the urgent need for collaborative efforts to address the challenges faced by the region. (BBC) (Al Jazeera)",
                "locations": "[{{'city': 'Tokyo', 'country': 'JP'}}]",
            },
            {
                "event": "In 2025, the political tension between Israel and Egypt escalated, leading to increased military presence in disputed territories. The conflict has drawn international attention, with multiple nations urging for peaceful negotiations. (Reuters) (CNN)",
                "locations": "[]",
            },
            {
                "event": "2006 World Cup - In the final, Italy defeats France to claim its first title in the event. (CNN)",
                "locations": "[]",
            },
            {
                "event": "Libyan Civil War - Forces aligned with the Government of National Accord launch an offensive in western Libya, capturing the cities of Sabratha, Surman, and Al-Ajaylat from forces loyal to Khalifa Haftar. (Al Jazeera)",
                "locations": "[{{'city': 'Sabratha', 'country': 'LY'}}, {{'city': 'Surman', 'country': 'LY'}}, {{'city': 'Al-Ajaylat', 'country': 'LY'}}]",
            },
            {
                "event": "Clashes break out and tear gas is fired by security forces in northern Azerbaijan after protesters torch the home of a local governor. (Daily Times of Pakistan)",
                "locations": "[{{'city': 'Baku', 'country': 'AZ'}}]",
            },
        ]

        example_formatter_template = """
        Event: {event}
        'locations': {locations}\n
        """

        example_prompt = PromptTemplate(
            input_variables=["event", "locations"],
            template=example_formatter_template,
        )

        prompt_template = """
        From the event description below, extract any specific cities in the format "city, alpha-2 country code" where the event clearly occurred. There can be multiple cities. Do not include regions, countries, or inferred locations. If a country is mentioned but no explicit city, return the common name of the capital city of the country. If no specific cities are mentioned, return {{ "locations": [] }}.

        {format_instructions}

        Event: {event_description}
        """

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Here are some examples of event descriptions and the place of occurrence associated with them:\n\n",
            suffix=prompt_template,
            example_separator="\n",
            input_variables=["event_description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        return few_shot_prompt | self.llm | self.parser

    def extract_locations_from_event(self, event_description: str):
        """Extract locations from an event description."""
        return self.chain.invoke(event_description).locations

    def check_location_in_database(self, location: Location, locations_data):
        """Check if the location exists in the database."""
        return location.country in locations_data and location.city in locations_data[location.country]

    def get_location_coordinates(self, location: Location):
        """Get coordinates for a city and country using OpenStreetMap."""
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{location.city},{location.country}",
            "format": "json",
            "limit": 1,  # Limit to the first result
        }
        headers = {"User-Agent": "YourAppName/1.0 (your_email@example.com)"}  # Replace with your app name and email
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            data = response.json()

            if response.status_code == 200 and data:
                return {
                    "lon": float(data[0]["lon"]),
                    "lat": float(data[0]["lat"]),
                }
            logging.debug(f"Error: Could not retrieve data for {location.city}, {location.country}")
            return None
        except Exception as e:  # pylint: disable=broad-except
            logging.debug(f"Error: {e}")
            return None

    def haversine(self, coord1, coord2):
        """Calculate the great-circle distance between two points on the Earth."""
        radius = 6371.0  # Radius of the Earth in km

        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = radius * c
        return distance

    def find_closest_city(self, target_coord, coord_data_cache):
        """Find the closest city to the given coordinates."""
        closest_city = None
        min_distance = float('inf')

        for coord, city_country in coord_data_cache.items():
            distance = self.haversine(target_coord, coord)
            if distance < min_distance:
                min_distance = distance
                closest_city = city_country

        return closest_city

    def get_locations_from_event(self, event_description):
        """Extract locations from an event description, validate them, and return the list of locations."""
        locations = self.extract_locations_from_event(event_description)

        if not locations:
            return []

        verified_locations = []
        for location in locations:
            if self.check_location_in_database(location, self.locations_data):
                verified_locations.append(location)
            else:
                coordinates = self.get_location_coordinates(location)
                if coordinates:
                    closest_city = self.find_closest_city((float(coordinates["lat"]), float(coordinates["lon"])), self.coord_data_cache)
                    if closest_city:
                        verified_locations.append(Location(city=closest_city[0], country=closest_city[1]))
        return verified_locations


class TitleExtractorManager:
    """Class to manage title extraction from event descriptions."""

    def __init__(self):
        """Initialize the TitleExtractorManager with required components."""
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
        self.parser = self._setup_parser()
        self.chain = self._setup_chain()

    def _setup_parser(self):
        """Set up the Pydantic output parser."""

        return PydanticOutputParser(pydantic_object=Title)

    def _setup_chain(self):
        """Set up the LLM chain with prompts and examples."""
        prompt_template = """
        From the event description below, extract a news headline that summarizes the event. The headline should be concise and capture the main event or incident described.

        {format_instructions}

        Event: {event_description}
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["event_description"], partial_variables={"format_instructions": self.parser.get_format_instructions()})

        return prompt | self.llm | self.parser

    def extract_title_from_event(self, event_description: str):
        """Extract a title from an event description."""
        return self.chain.invoke(event_description).title


def main():
    """Main function to test the LocationExtractionManager."""
    manager = LocationExtractionManager()

    events = [
        "2017 Westminster attack - An attacker carries out a vehicle-ramming attack outside the Houses of Parliament in London, before stabbing a police officer and subsequently being shot by security forces. Including the attacker, there are at least four people dead. (BBC) (Sky News) (Fox News)",
        "Boko Haram insurgency - 2017 Maiduguri attack - Multiple suicide blasts occur at a refugee camp near Maiduguri, Nigeria, leaving eight people dead and wounding 18 others. (Al Jazeera) (All Africa)",
        "March 2017 Israel–Syria incident - The Israeli Air Force resumes bombing against suspected Hezbollah-related targets in Syria, with a fourth round of bombings occurring near military sites near Damascus. (The Financial Times)",
        "Eurovision Song Contest 2017, Russia–Ukraine relations - The Security Service of Ukraine bans Yuliya Samoylova, Russia's entrant in this year's Eurovision Song Contest, from entering the country. This follows allegations that she had toured in Crimea following its annexation by Russia. (The Washington Post)",
        "Insurgency in the North Caucasus - Six Chechen soldiers, along with six militants, are killed in a nighttime attack on a Russian National Guard base in Chechnya. (BBC)",
        "The Olympic Games went well this year!",
        "Syrian Civil War - Syrian rebels, led by Tahrir al-Sham, launch an offensive in the Hama Governorate, capturing the towns of Suran, Khitab and Al-Majdal. (Reuters)",
        "The surfing competition in France was a great success!",
        "2017 World Baseball Classic - In the final, the United States defeats Puerto Rico 8–0 to claim its first title in the event. Toronto Blue Jays starting pitcher Marcus Stroman, who threw six hitless innings in the final for Team USA, is named tournament MVP. (AP via ESPN)",
    ]

    for event_description in events:
        locations = manager.get_locations_from_event(event_description)
        logging.info(locations)


if __name__ == "__main__":
    main()
