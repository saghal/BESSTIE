"""
Filename: google_places.py

This module interacts with the Google Places API to:
    - Search for places near specific city coordinates based on a set of place types.
    - Retrieve review data for these places.
    - Aggregate reviews across multiple cities and save them into CSV files for further analysis.

Prerequisites:
    - Set up a valid Google Places API key and store it in an environment variable named 'GOOGLE_PLACES_KEY'.
    - CSV files containing city names and coordinates should be available in the designated data directory.
    - A JSON file listing the place types is needed (curated from Google's supported types).

Usage:
    1. Ensure that you have the required CSV files for cities (e.g., "en-AU.csv", "en-IN.csv", "en-UK.csv")
       in the folder './Dataset/Cities/'.
    2. Ensure that the JSON file "place_types.json" is available under './Dataset/'.
    3. Run the script. It will create review CSV files under './Dataset/Raw Data/Google Reviews/' for each region.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocationCoordinates(BaseModel):
    """
    Represents geographic coordinates with latitude and longitude.

    Attributes:
        latitude (float): The latitude coordinate.
        longitude (float): The longitude coordinate.
    """

    latitude: float
    longitude: float

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate that latitude is between -90 and 90 degrees."""
        if v < -90 or v > 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate that longitude is between -180 and 180 degrees."""
        if v < -180 or v > 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {v}")
        return v


class CircleArea(BaseModel):
    """
    Represents a circular area defined by a center point and radius.

    Attributes:
        center (LocationCoordinates): The center coordinates of the circle.
        radius (float): The radius of the circle in meters.
    """

    center: LocationCoordinates
    radius: float = Field(500.0, description="Radius in meters")

    @field_validator("radius")
    @classmethod
    def validate_radius(cls, v: float) -> float:
        """Validate that radius is positive."""
        if v <= 0:
            raise ValueError(f"Radius must be positive, got {v}")
        return v


class LocationRestriction(BaseModel):
    """
    Defines a geographic restriction for searching places.

    Attributes:
        circle (CircleArea): The circular area that defines the search boundary.
    """

    circle: CircleArea


class PlaceSearchParams(BaseModel):
    """
    Parameters for the Google Places API search request.

    Attributes:
        locationRestriction (LocationRestriction): The geographic area to search within.
        includedTypes (List[str]): Types of places to include in the search results.
        maxResultCount (int): Maximum number of results to return.
    """

    locationRestriction: LocationRestriction
    includedTypes: List[str]
    maxResultCount: int = Field(20, description="Limits the number of places returned")

    @field_validator("maxResultCount")
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        """Validate that maxResultCount is between 1 and 20."""
        if v < 1 or v > 20:
            raise ValueError(f"maxResultCount must be between 1 and 20, got {v}")
        return v


class AuthorAttribution(BaseModel):
    """
    Information about the author of a review.

    Attributes:
        uri (AnyHttpUrl): The URI associated with the author.
        displayName (Optional[str]): The display name of the author.
    """

    uri: AnyHttpUrl
    displayName: Optional[str] = None


class ReviewText(BaseModel):
    """
    Contains the text content of a review.

    Attributes:
        text (str): The actual text of the review.
        languageCode (Optional[str]): The language code of the review text.
    """

    text: str
    languageCode: Optional[str] = None


class Review(BaseModel):
    """
    Represents a review from Google Places API.

    Attributes:
        authorAttribution (AuthorAttribution): Information about the review author.
        rating (int): The rating given in the review (1-5).
        originalText (ReviewText): The original text of the review.
        publishTime (Optional[str]): When the review was published.
    """

    authorAttribution: AuthorAttribution
    rating: int
    originalText: ReviewText
    publishTime: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        """Validate that rating is between 1 and 5."""
        if v < 1 or v > 5:
            raise ValueError(f"Rating must be between 1 and 5, got {v}")
        return v


class PlaceId(TypedDict):
    """
    Represents a place identifier returned by the Google Places API.

    Attributes:
        id (str): The unique identifier for a place.
    """

    id: str


class ReviewsData(TypedDict):
    """
    Structure for storing aggregated review data.

    Attributes:
        id (List[str]): List of reviewer identifiers.
        rating (List[int]): List of ratings given by reviewers.
        review (List[str]): List of review texts.
        city (List[str]): List of city names where the reviews were collected.
    """

    id: List[str]
    rating: List[int]
    review: List[str]
    city: List[str]


def search_places(location: str, types: List[str]) -> List[Dict[str, Any]]:
    """
    Searches for nearby places using Google Places API based on a specific location and a list of types.

    The function splits the provided location into latitude and longitude, constructs a circular
    location restriction, and queries the Places API for nearby places that match the provided
    types. It returns a list of places in JSON format, where each place dictionary contains only
    the Place ID.

    Args:
        location: The geographic coordinates of a city in the format "latitude,longitude".
                 (Coordinates can be obtained for example from https://simplemaps.com/data/world-cities)
        types: A list of place types to filter the search results.
               (Types are curated from https://developers.google.com/maps/documentation/places/web-service/supported_types)

    Returns:
        A list of dictionaries for each found place, each containing the key 'id'.
        If the response cannot be decoded as JSON, it returns a list with an error message.

    Raises:
        ValueError: If the location format is invalid or API key is missing.
    """
    api_key = os.getenv("GOOGLE_PLACES_KEY")
    if not api_key:
        raise ValueError("Google Places API key not found in environment variables")

    try:
        lat_str, lng_str = location.split(",")
        latitude = float(lat_str)
        longitude = float(lng_str)
    except ValueError:
        raise ValueError(
            f"Invalid location format: {location}. Expected 'latitude,longitude'"
        )

    # Create location coordinates with validation
    coordinates = LocationCoordinates(latitude=latitude, longitude=longitude)

    # Create circle area
    circle = CircleArea(center=coordinates)

    # Create location restriction
    location_restriction = LocationRestriction(circle=circle)

    # Create search parameters
    params = PlaceSearchParams(
        locationRestriction=location_restriction, includedTypes=types
    )

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id",
    }

    try:
        response = requests.post(url, headers=headers, json=params.dict())
        response.raise_for_status()  # Raise exception for HTTP errors

        response_data = response.json()
        return response_data.get("places", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from response: {response.text}")
        return []


def get_fields(place_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves review data for a specific place identified by its Place ID using the Google Places API.

    The function sends a GET request to the API endpoint for a given Place ID and requests only the reviews field.
    It then extracts and returns a list of review objects from the resulting JSON data.

    Args:
        place_id: The unique identifier of a place on Google Maps.

    Returns:
        A list of dictionaries where each dictionary represents a review of the place.
        If no reviews are found, an empty list is returned.

    Raises:
        ValueError: If the API key is missing.
    """
    api_key = os.getenv("GOOGLE_PLACES_KEY")
    if not api_key:
        raise ValueError("Google Places API key not found in environment variables")

    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "reviews",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        data = response.json()
        return data.get("reviews", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for place {place_id}: {str(e)}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from response for place {place_id}")
        return []


def get_reviews(cities_df: pd.DataFrame, types: List[str]) -> ReviewsData:
    """
    Collects reviews from places across multiple cities and returns them in a structured dictionary.

    For each city (each row in the provided DataFrame), the function:
      1. Searches for nearby places matching the provided types.
      2. Retrieves reviews for each place.
      3. Extracts key review details such as the review's author (using a parsed version from the URI),
         the rating, the original review text, and the city name.
      4. Aggregates these details into a dictionary with lists for each field.

    Args:
        cities_df: A DataFrame that contains at least two columns:
                   - 'city': Name of the city.
                   - 'location': A string of the city's coordinates formatted as "latitude,longitude".
        types: A list of place types to be searched for each city.

    Returns:
        A dictionary with the following keys:
        - "id": List of author identifiers (parsed from their URI).
        - "rating": List of ratings provided by the reviewers.
        - "review": List of the original review texts.
        - "city": List of corresponding city names from which the reviews were retrieved.
    """
    # Validate required columns in DataFrame
    required_columns = ["city", "location"]
    for col in required_columns:
        if col not in cities_df.columns:
            raise ValueError(f"Required column '{col}' not found in cities DataFrame")

    reviews_data: ReviewsData = {"id": [], "rating": [], "review": [], "city": []}

    for _, row in tqdm(
        cities_df.iterrows(), total=cities_df.shape[0], desc="Processing Cities"
    ):
        city_name = row["city"]
        location = row["location"]

        logger.info(f"Searching for places in {city_name}")
        places = search_places(location, types)

        for place in places:
            place_id = place.get("id", "")
            if not place_id:
                logger.warning(f"Skipping place with no ID in {city_name}")
                continue

            logger.info(f"Getting reviews for place {place_id} in {city_name}")
            reviews = get_fields(place_id)

            for review_data in reviews:
                try:
                    # Validate and process the review data
                    review = Review(
                        authorAttribution=review_data.get("authorAttribution", {}),
                        rating=review_data.get("rating", 0),
                        originalText=review_data.get("originalText", {"text": "N/A"}),
                    )

                    # Extract the unique identifier from the author attribution's URI
                    author_id = str(review.authorAttribution.uri).split("/")[-2]

                    reviews_data["id"].append(author_id)
                    reviews_data["rating"].append(review.rating)
                    reviews_data["review"].append(review.originalText.text)
                    reviews_data["city"].append(city_name)
                except Exception as e:
                    logger.error(
                        f"Error processing review for place {place_id}: {str(e)}"
                    )
                    continue

    return reviews_data


def main() -> None:
    """
    Main function to execute the review collection and saving process.

    This function:
    1. Loads environment variables
    2. Reads city data from CSV files
    3. Loads place types from a JSON file
    4. Collects reviews for each region
    5. Saves the reviews to CSV files

    Returns:
        None
    """
    # Load environment variables from a .env file; ensure that the GOOGLE_PLACES_KEY is set.
    load_dotenv()

    if not os.getenv("GOOGLE_PLACES_KEY"):
        logger.error("GOOGLE_PLACES_KEY environment variable not set")
        return

    # Define the base directory for raw data files.
    data_dir = Path("./Dataset")
    cities_dir = data_dir / "Cities"
    output_dir = data_dir / "Raw Data" / "Google Reviews"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load place types
    place_types_path = data_dir / "place_types.json"
    try:
        with open(place_types_path, "r") as file:
            types = json.load(file)

        if not isinstance(types, list):
            logger.error("place_types.json does not contain a list")
            return
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading place types: {str(e)}")
        return

    # Process regions
    regions = ["en-AU", "en-UK", "en-IN"]

    for region in regions:
        try:
            # Read city data CSV file
            city_file_path = cities_dir / f"{region}.csv"
            logger.info(f"Reading city data from {city_file_path}")

            if not city_file_path.exists():
                logger.error(f"City file not found: {city_file_path}")
                continue

            cities_df = pd.read_csv(city_file_path)

            # Collect reviews
            logger.info(f"Collecting reviews for {region}")
            reviews = get_reviews(cities_df, types)

            # Save to CSV
            output_file_path = output_dir / f"{region}.csv"
            logger.info(f"Saving reviews to {output_file_path}")
            pd.DataFrame(reviews).to_csv(output_file_path, index=False)

            logger.info(f"Successfully processed {region}")
        except Exception as e:
            logger.error(f"Error processing region {region}: {str(e)}")


if __name__ == "__main__":
    main()
