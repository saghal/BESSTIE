"""
File name: reddit_threads.py

This module authenticates with the Reddit API and retrieves posts from multiple subreddits defined
in an external JSON configuration file. The script performs the following tasks:

1. Loads environment variables from a .env file containing Reddit API credentials.
2. Authenticates with Reddit to obtain an access token.
3. Loads subreddit lists from the specified JSON configuration file (Utilities/subreddits.json).
4. Fetches hot posts from each subreddit.
5. Extracts selected fields from the posts.
6. Combines all the extracted data into a Pandas DataFrame and exports the data to a CSV file.

Prerequisites:
    - A .env file with the following environment variables:
        - REDDIT_CLIENT: Your Reddit client ID.
        - REDDIT_SECRET: Your Reddit client secret.
        - REDDIT_USER: Your Reddit username.
        - REDDIT_PASS: Your Reddit password.
    - Python packages: requests, pandas, python-dotenv, pydantic.
    - A JSON file (Utilities/subreddits.json) containing subreddit configurations as described above.

Usage:
    Run this script directly. It will authenticate with Reddit, read subreddit configuration, fetch posts,
    and save the data to "{locale}.csv".
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RedditCredentials(BaseModel):
    """
    Represents the credentials required for Reddit API authentication.

    Attributes:
        client_id (str): Reddit API client ID.
        client_secret (str): Reddit API client secret.
        username (str): Reddit username.
        password (str): Reddit password.
    """

    client_id: str = Field(..., description="Reddit API client ID")
    client_secret: str = Field(..., description="Reddit API client secret")
    username: str = Field(..., description="Reddit username")
    password: str = Field(..., description="Reddit password")

    @classmethod
    def from_env(cls) -> "RedditCredentials":
        """
        Creates a RedditCredentials instance from environment variables.

        Returns:
            RedditCredentials: An instance with credentials from environment variables.

        Raises:
            ValueError: If any required environment variable is missing.
        """
        try:
            return cls(
                client_id=os.environ["REDDIT_CLIENT"],
                client_secret=os.environ["REDDIT_SECRET"],
                username=os.environ["REDDIT_USER"],
                password=os.environ["REDDIT_PASS"],
            )
        except KeyError as e:
            raise ValueError(f"Missing environment variable: {e}")


class RedditAuthToken(BaseModel):
    """
    Represents a Reddit API authentication token.

    Attributes:
        access_token (str): The access token for Reddit API.
        token_type (str): The type of token, typically "bearer".
        expires_in (int): Token expiration time in seconds.
    """

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(3600, description="Token expiration time in seconds")

    def get_authorization_header(self) -> str:
        """
        Creates the authorization header value using the token.

        Returns:
            str: The authorization header value.
        """
        return f"{self.token_type} {self.access_token}"


class RequestHeaders(BaseModel):
    """
    Represents HTTP headers for Reddit API requests.

    Attributes:
        user_agent (str): User agent string identifying the application.
        authorization (Optional[str]): Authorization header value if authenticated.
    """

    user_agent: str = "MyAPI/0.0.1"
    authorization: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """
        Converts the headers to a dictionary format for requests.

        Returns:
            Dict[str, str]: Dictionary of headers.
        """
        headers = {"User-Agent": self.user_agent}
        if self.authorization:
            headers["Authorization"] = self.authorization
        return headers


class SubredditConfig(TypedDict):
    """
    Type definition for subreddit configuration structure.

    Example:
        {
            "en-AU": ["australia", "sydney", "melbourne"],
            "en-UK": ["unitedkingdom", "london", "casualuk"]
        }
    """

    pass


class RedditPost(BaseModel):
    """
    Represents a Reddit post with selected fields.

    Attributes:
        subreddit (str): The subreddit name.
        title (str): The post title.
        selftext (str): The post content text.
        upvote_ratio (float): Ratio of upvotes to total votes.
        ups (int): Number of upvotes.
        downs (int): Number of downvotes.
        score (int): Post score (difference between upvotes and downvotes).
        locale (Optional[str]): Optional locale identifier added during processing.
    """

    subreddit: str
    title: str
    selftext: str = ""
    upvote_ratio: float = Field(0.0, ge=0.0, le=1.0)
    ups: int = 0
    downs: int = 0
    score: int = 0
    locale: Optional[str] = None

    @model_validator(mode="after")
    def validate_score(self) -> "RedditPost":
        """
        Validates that score is consistent with ups and downs.

        Returns:
            The validated RedditPost instance.
        """
        # In newer Reddit API, downs is usually 0 and score equals ups
        # This is a soft validation that doesn't raise an error
        if self.downs == 0 and self.score != self.ups:
            logger.warning(
                f"Inconsistent score: ups={self.ups}, downs={self.downs}, score={self.score}"
            )

        return self


def get_reddit_token() -> RedditAuthToken:
    """
    Authenticates with Reddit and obtains an access token using credentials from environment variables.

    Returns:
        RedditAuthToken: A valid Reddit API access token model.

    Raises:
        ValueError: If credentials are missing from environment variables.
        requests.HTTPError: If the request to the Reddit API fails.
        ValueError: If the access token cannot be obtained from the API.
    """
    try:
        # Load credentials from environment variables
        credentials = RedditCredentials.from_env()

        # Set up HTTP Basic Authentication using Reddit API client credentials
        auth = requests.auth.HTTPBasicAuth(
            credentials.client_id, credentials.client_secret
        )

        # Define the payload with login details needed to request an access token
        login_data = {
            "grant_type": "password",
            "username": credentials.username,
            "password": credentials.password,
        }

        # Set headers for authentication request
        headers = RequestHeaders().to_dict()

        # Request an access token from Reddit
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data=login_data,
            headers=headers,
        )
        response.raise_for_status()

        # Parse response into token model
        token_data = response.json()
        if "access_token" not in token_data:
            raise ValueError(
                "Failed to obtain Reddit access token: No access_token in response"
            )

        return RedditAuthToken(**token_data)

    except KeyError as e:
        raise ValueError(f"Missing environment variable: {e}")
    except requests.HTTPError as e:
        logger.error(f"HTTP error during authentication: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}")
        raise ValueError(f"Failed to obtain Reddit access token: {e}")


def construct_headers(token: RedditAuthToken) -> Dict[str, str]:
    """
    Constructs headers for authenticated requests to Reddit.

    Args:
        token: The Reddit API access token model.

    Returns:
        A dictionary of HTTP headers including the Authorization token.
    """
    headers = RequestHeaders(authorization=token.get_authorization_header())
    return headers.to_dict()


def fetch_subreddit_posts(
    subreddit: str, headers: Dict[str, str], limit: int = 100
) -> Dict[str, Any]:
    """
    Fetches hot posts from a specified subreddit.

    Args:
        subreddit: The subreddit name from which to fetch posts.
        headers: The HTTP headers for authentication and User-Agent.
        limit: The maximum number of posts to retrieve. Defaults to 100.

    Returns:
        The JSON response from Reddit containing the posts data.

    Raises:
        requests.HTTPError: If the request to the Reddit API fails.
        ValueError: If the response cannot be parsed as JSON.
    """
    if limit < 1 or limit > 100:
        logger.warning(f"Invalid limit {limit}, using default value of 100")
        limit = 100

    url = f"https://oauth.reddit.com/r/{subreddit}/hot"
    params = {"limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logger.error(f"Failed to fetch posts from r/{subreddit}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from r/{subreddit}: {e}")
        raise ValueError(f"Invalid JSON response from Reddit API: {e}")


def parse_posts(posts_json: Dict[str, Any]) -> List[RedditPost]:
    """
    Parses the JSON response from Reddit to extract selected post data.

    Args:
        posts_json: The JSON response from the Reddit API for posts.

    Returns:
        A list of RedditPost models, each containing data of a single post.
    """
    posts_list = []

    # Check for expected structure
    if "data" not in posts_json or "children" not in posts_json.get("data", {}):
        logger.warning("Unexpected JSON structure in Reddit API response")
        return []

    for post in posts_json.get("data", {}).get("children", []):
        post_data = post.get("data", {})

        try:
            # Create RedditPost model with validation
            post_model = RedditPost(
                subreddit=post_data.get("subreddit", "unknown"),
                title=post_data.get("title", ""),
                selftext=post_data.get("selftext", ""),
                upvote_ratio=post_data.get("upvote_ratio", 0.0),
                ups=post_data.get("ups", 0),
                downs=post_data.get("downs", 0),
                score=post_data.get("score", 0),
            )
            posts_list.append(post_model)
        except Exception as e:
            logger.warning(f"Failed to parse post: {e}")
            continue

    return posts_list


def load_subreddit_config(filepath: Union[str, Path]) -> SubredditConfig:
    """
    Loads subreddit configuration from a JSON file.

    Args:
        filepath: Path to the JSON file containing the subreddit configuration.

    Returns:
        A dictionary mapping locale keys (e.g., en-AU, en-IN, en-UK) to lists of subreddit names.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        ValueError: If the JSON structure doesn't match the expected format.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Subreddit configuration file not found: {filepath}")

    try:
        with open(filepath, "r") as file:
            config = json.load(file)

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Subreddit configuration must be a dictionary")

        for locale, subreddits in config.items():
            if not isinstance(subreddits, list):
                raise ValueError(f"Subreddits for locale '{locale}' must be a list")

        return config
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in subreddit configuration file: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading subreddit configuration: {e}")
        raise ValueError(f"Failed to load subreddit configuration: {e}")


def main() -> None:
    """
    Main function to drive the Reddit API fetcher workflow:
        1. Loads environment variables and retrieves the Reddit API token.
        2. Constructs HTTP headers for authenticated API requests.
        3. Reads subreddit configuration from the JSON file.
        4. Iterates over each locale and its list of subreddits:
           a. Fetches hot posts for each subreddit.
           b. Parses and augments the post data with a locale identifier.
           c. Aggregates all posts into a single list.
        5. Converts the aggregated data into a Pandas DataFrame and exports it as a CSV file.
    """
    # Load environment variables from the .env file
    load_dotenv()
    logger.info("Environment variables loaded")

    try:
        # Obtain Reddit access token
        token = get_reddit_token()
        logger.info("Successfully authenticated with Reddit API")

        # Construct HTTP headers for subsequent API requests
        headers = construct_headers(token)

        # Load subreddit configuration from the JSON file
        config_path = Path("Utilities") / "subreddits.json"
        subreddit_config = load_subreddit_config(config_path)
        logger.info(
            f"Loaded subreddit configuration with {len(subreddit_config)} locales"
        )

        # Iterate over each locale and its respective subreddits
        for locale, subreddits in subreddit_config.items():
            logger.info(
                f"Processing locale: {locale} with {len(subreddits)} subreddits"
            )
            all_posts: List[Dict[str, Any]] = (
                []
            )  # Will store all posts fetched from all in a locale subreddits

            for subreddit in subreddits:
                logger.info(f"Fetching posts from r/{subreddit}")

                try:
                    # Fetch posts from the subreddit
                    posts_json = fetch_subreddit_posts(subreddit, headers)
                    posts = parse_posts(posts_json)

                    logger.info(f"Retrieved {len(posts)} posts from r/{subreddit}")

                    # Add locale information and convert to dictionaries
                    for post in posts:
                        post.locale = locale
                        all_posts.append(post.dict())

                except Exception as e:
                    logger.error(f"Error processing r/{subreddit}: {e}")
                    continue

            # Create a DataFrame from the combined posts data
            if all_posts:
                df = pd.DataFrame(all_posts)
                output_filepath = Path("Dataset") / "Reddit" / f"{locale}.csv"
                df.to_csv(output_filepath, index=False)
                logger.info(f"Saved {len(all_posts)} posts to {output_filepath}")
            else:
                logger.warning("No posts were fetched from the specified subreddits")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        return


if __name__ == "__main__":
    main()
