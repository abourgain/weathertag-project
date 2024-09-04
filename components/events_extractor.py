"""Event extraction from Wikipedia's Current Events page."""

import requests
import streamlit as st
from bs4 import (
    BeautifulSoup,
)

from components.llm import LocationExtractionManager, TitleExtractorManager
from components.location_embeddings import LocationEmbeddingManager  # See: https://www.crummy.com/software/BeautifulSoup/bs4/doc/


@st.cache_data
def wikipedia_pageprops(href):
    """Fetch page properties from Wikipedia API."""
    if href.startswith("/wiki"):
        wp_title = href.removeprefix("/wiki/")
        wp_api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&format=json&titles={wp_title}"
        res = requests.get(wp_api_url, timeout=10)
        for page in res.json()["query"].get("pages", {}).values():
            return page.get("pageprops", {})
    return {}


def n_events_old_extractor(html, **kwargs):
    """
    Return the number of events extracted with the previous method
    """

    soup = BeautifulSoup(html, features='html.parser')
    n_events = 0
    for _ in soup.select(".current-events-content ul li"):
        n_events += 1
    return n_events


def computer_event_duplication_rate(n_old, n_current):
    """Compute the rate of event duplication."""
    return (n_old - n_current) / n_current


def extract_event_details(
    date,
    category,
    event_html,
    location_manager: LocationExtractionManager | LocationEmbeddingManager | None = None,
    title_manager: TitleExtractorManager | None = None,
):  # pylint: disable=too-many-locals
    """
    Extracts the title, content, direct event link, wiki links, news links, and other details from an event.
    """
    soup = BeautifulSoup(event_html, features='html.parser')

    title_tag = soup.find('a')
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # Extract the direct event link
    direct_event_link = title_tag['href'] if title_tag and title_tag.has_attr('href') else None
    if direct_event_link:
        direct_event_link = f"https://en.wikipedia.org{direct_event_link}"

    # Initialize dictionaries for wiki links and other links
    wiki_links = {}
    other_links = {}

    # Extract content: Replace <a> tags with their text content, adding spaces where needed
    for tag in soup.find_all('a'):
        href = tag['href'] if tag.has_attr('href') else None
        text = tag.get_text(strip=True)

        if href:
            if href.startswith('/wiki/'):
                # Add to wiki links dictionary
                wiki_links[text] = href
            else:
                # Add to other links dictionary
                text = text.replace('(', '').replace(')', '')
                other_links[text] = href

            # Unwrap the tag to keep the text but remove the link
            tag.unwrap()

    # Extract the remaining content as text
    content = soup.get_text()[3:].strip()  # Remove the leading " - " and strip whitespace

    # Title extraction from the content
    if title_manager:
        title = title_manager.extract_title_from_event(content)

    event_details = {
        'title': title,
        'content': content,
        'date': str(date),
        'category': category,
        'direct_event_link': direct_event_link,
        'wiki_links': wiki_links,
        'other_links': other_links,
    }

    # Find the closest locations to the event
    if isinstance(location_manager, LocationEmbeddingManager):
        locations = location_manager.find_closest_locations(content, top_k=5)
        event_details['locations'] = locations
    elif isinstance(location_manager, LocationExtractionManager):
        locations = location_manager.get_locations_from_event(content)
        event_details['locations'] = locations
    else:
        pass

    return event_details


def extract_current_events(
    html,
    dt,
    location_manager: LocationExtractionManager | LocationEmbeddingManager | None = None,
    title_manager: TitleExtractorManager | None = None,
    **kwargs,
):
    """
    Extract events from a Wikipedia page HTML, considering categories and nested lists.
    Concatenate HTML contents of each event while removing <ul> and <li> tags.
    The higher-level HTML content is included in each sub-event.
    Yields event details including title, content, direct event link, and other links.
    """
    soup = BeautifulSoup(html, features='html.parser')

    current_category = None

    def extract_events(li_element, parent_content=""):
        """
        Recursively extract events from nested <ul> lists, concatenating their HTML content,
        and then extract event details.
        """
        # Extract content before the first nested <ul> (if any)
        content_before_ul = ''
        nested_ul = None

        for child in li_element.children:
            if child.name == 'ul':
                nested_ul = child
                break
            content_before_ul += str(child)

        # Combine parent content and current content
        combined_content = parent_content + ' - ' + content_before_ul.strip()

        if nested_ul:
            nested_lis = nested_ul.find_all("li", recursive=False)
            for nested_li in nested_lis:
                yield from extract_events(nested_li, combined_content)
        else:
            # No nested <ul>, this is a complete event
            cleaned_content = remove_ul_li_tags(combined_content)
            yield extract_event_details(dt, current_category, cleaned_content, location_manager, title_manager)

    def remove_ul_li_tags(html_content):
        """
        Remove <ul> and <li> tags from the HTML content.
        """
        soup = BeautifulSoup(html_content, features='html.parser')
        for tag in soup.find_all(['ul', 'li']):
            tag.unwrap()  # Remove the tag but keep its content
        return str(soup)

    # Find all categories and their corresponding events
    categories = soup.find_all("div", class_="current-events-content-heading")
    for category in categories:
        current_category = category.get_text(strip=True)
        # The next sibling after a category should be a <ul> containing events
        next_sibling = category.find_next_sibling("ul")
        if next_sibling:
            top_level_lis = next_sibling.find_all("li", recursive=False)
            for li in top_level_lis:
                yield from extract_events(li)
