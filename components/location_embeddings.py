"""Python module for generating location embeddings and finding the closest locations to a given event text."""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
import streamlit as st

from components.llm import Location
from components.weather import get_country_name, load_weather_data


@st.cache_resource
def load_model(model_name):
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name, trust_remote_code=True)


@st.cache_data
def load_locations(file_path: str = "./data/locations.json", save_local: bool = False):
    """Load the locations from a json file and return them as a list."""
    if save_local and Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            locations = json.load(f)
    else:
        _, locations, _ = load_weather_data(save_local=save_local)

    location_country_codes = [f"{city}, {country_code}" for country_code, cities in locations.items() for city in cities]
    location_country_names = [f"{city}, {get_country_name(country_code)}" for country_code, cities in locations.items() for city in cities]

    return location_country_codes, location_country_names


@st.cache_data
def compute_location_embeddings(location_names, model_name):
    """Compute embeddings for locations and return them."""
    model = load_model(model_name)
    location_embeddings = model.encode(location_names, show_progress_bar=False, convert_to_tensor=True)
    return location_embeddings.cpu().numpy()  # Convert to NumPy array


class LocationEmbeddingManager:
    """Class for managing location embeddings and finding the closest locations to a given event text."""

    def __init__(
        self,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_path="./models/nomic-embed-text-v1.5",
        location_embeddings_path: str = "./data/location_embeddings.npy",
        location_names_path: str = "./data/locations.json",
        save_local: bool = False,
    ):
        """Initialize the LocationEmbeddingManager with the specified model."""
        self.save_local = save_local
        self.model_path = Path(model_path)
        self.model_name = model_name

        self.model = load_model(self.model_name)
        self.location_country_codes, self.location_country_names = load_locations(file_path=location_names_path, save_local=self.save_local)
        self.location_embeddings = self.load_location_embeddings(file_path=location_embeddings_path)

    def load_location_embeddings(self, file_path: str = "./data/location_embeddings.npy"):
        """Load embeddings for locations from a .npy file or compute them if necessary."""
        if self.save_local and Path(file_path).exists():
            location_embeddings = np.load(file_path)
        else:
            location_embeddings = compute_location_embeddings(self.location_country_names, self.model_name)
            if self.save_local:
                np.save(file_path, location_embeddings)  # Save as a .npy file

        return location_embeddings

    def embed_text(self, texts):
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_tensor=True)
        return embeddings

    def embed_event(self, event_text):
        """Generate an embedding for a given event text."""
        event_embedding = self.embed_text([event_text])
        return event_embedding

    def find_closest_locations(self, event_text, top_k=5):
        """Find the closest locations to the given event text."""
        # Embed the event text
        event_embedding = self.embed_event(event_text)

        # Move tensors to CPU and convert to NumPy arrays for cosine similarity
        event_embedding_cpu = event_embedding.cpu().numpy()
        if isinstance(self.location_embeddings, torch.Tensor):
            location_embeddings_cpu = self.location_embeddings.cpu().numpy()
        else:
            location_embeddings_cpu = self.location_embeddings

        # Compute cosine similarity between event embedding and all location embeddings
        similarities = cosine_similarity(event_embedding_cpu, location_embeddings_cpu)

        # Get the indices of the top_k most similar locations
        closest_indices = np.argsort(similarities[0])[::-1][:top_k]

        # Get the corresponding location names in the Location format
        closest_location_codes = [self.location_country_codes[idx] for idx in closest_indices]
        closest_locations = [Location(city=location.split(",")[0], country=location.split(",")[1].strip()) for location in closest_location_codes]

        return closest_locations


def main():
    """Main function for testing the LocationEmbeddingManager."""
    manager = LocationEmbeddingManager()
    content = "The weather in New York City is nice today."
    logging.info(manager.find_closest_locations(content, top_k=5))


if __name__ == "__main__":
    main()
