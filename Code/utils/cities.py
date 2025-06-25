"""
cities.py

This script processes world cities data from a CSV file, filters it by country and population,
and extracts city names with their coordinates for Australia, India, and the United Kingdom.
The filtered data is saved to country-specific CSV files.

Requirements:
    - pandas
    - pathlib
    - A world_cities.csv file in the ./Utilities/Cities/ directory containing columns:
      'city', 'country', 'lat', 'lng', and 'population'
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_location_string(row: pd.Series) -> str:
    """
    Create a location string from latitude and longitude.

    Args:
        row: DataFrame row containing 'lat' and 'lng' columns

    Returns:
        A string in the format "latitude,longitude"
    """
    return f"{row['lat']},{row['lng']}"


def filter_cities(df: pd.DataFrame, country: str, min_population: int) -> pd.DataFrame:
    """
    Filter cities dataframe by country and minimum population.

    Args:
        df: DataFrame containing city data
        country: Country name to filter by
        min_population: Minimum population threshold

    Returns:
        Filtered DataFrame with reset index
    """
    filtered_df = df.loc[
        (df["country"] == country) & (df["population"] > min_population)
    ].reset_index(drop=True)

    logger.info(
        f"Filtered {len(filtered_df)} cities from {country} with population > {min_population}"
    )
    return filtered_df


def process_and_save_cities(
    cities_df: pd.DataFrame, output_dir: Path, country_configs: List[Dict[str, Any]]
) -> None:
    """
    Process cities data for multiple countries and save to CSV files.

    Args:
        cities_df: DataFrame containing all cities data
        output_dir: Directory to save output CSV files
        country_configs: List of dictionaries with filtering parameters
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in country_configs:
        country = config["country"]
        min_pop = config["min_population"]
        file_code = config["file_code"]

        try:
            # Filter cities by country and population
            filtered_cities = filter_cities(cities_df, country, min_pop)

            # Add location column
            filtered_cities["location"] = filtered_cities.apply(
                create_location_string, axis=1
            )

            # Select and save required columns
            output_file = output_dir / f"{file_code}.csv"
            filtered_cities[["city", "location"]].to_csv(output_file, index=False)

            logger.info(f"Saved {len(filtered_cities)} cities to {output_file}")

        except Exception as e:
            logger.error(f"Error processing {country} data: {str(e)}")


def main() -> None:
    """Main function to process world cities data."""
    try:
        # Define data directory path
        data_dir = Path("./Utilities/")
        cities_path = data_dir / "Cities" / "world_cities.csv"
        output_dir = data_dir / "Cities"

        # Check if input file exists
        if not cities_path.exists():
            logger.error(f"Cities data file not found: {cities_path}")
            return

        # Read cities data
        logger.info(f"Reading cities data from {cities_path}")
        cities = pd.read_csv(cities_path)

        # Define country configurations
        country_configs = [
            {"country": "Australia", "min_population": 10000, "file_code": "en-AU"},
            {"country": "India", "min_population": 100000, "file_code": "en-IN"},
            {
                "country": "United Kingdom",
                "min_population": 50000,
                "file_code": "en-UK",
            },
        ]

        # Process and save city data
        process_and_save_cities(cities, output_dir, country_configs)

        logger.info("City data processing completed successfully")

    except Exception as e:
        logger.error(f"Error processing cities data: {str(e)}")


if __name__ == "__main__":
    main()
