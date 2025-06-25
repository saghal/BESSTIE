"""
Module: data_processing.py

This module processes annotated data from Excel files and produces training, validation, and test splits
for NLP tasks such as sentiment and sarcasm classification. The script carries out the following steps:

1. Reads review Excel files for each specified location from the annotated data directory.
2. Iterates over review domains ("Google" and "Reddit") within each file.
3. Preprocesses the review text and converts ratings into binary labels using a custom Preprocess class.
4. Filters out reviews with a sentiment label equal to 2, keeping only valid reviews.
5. For each classification task (sentiment and sarcasm):
    a. Selects the required columns: 'id', 'text', and the task-specific label.
    b. Renames the task-specific label to 'label' for consistency.
    c. Splits the data into training (70%), validation (10%), and test (20%) subsets using stratified sampling.
6. Saves each of the resulting split datasets into CSV files arranged in a nested directory structure:
       <split_path>/<Task>/<en-LOCATION>/<Domain>/[train.csv, valid.csv, test.csv]

Prerequisites:
    - The annotated review data must be stored in "./Dataset/Annotated Data/".
    - Excel files should be named "en-<LOCATION>.xlsx", where <LOCATION> is the uppercase code for the location (e.g., "en-AU.xlsx").
    - Each Excel file should contain sheets named "Google" and "Reddit".
    - A custom module 'utils/data_util.py' must be present with a Preprocess class used for text preprocessing and label conversion.
    - Required Python packages: os, pathlib, pandas, scikit-learn, tqdm, pydantic, and openpyxl (for reading Excel files).

Usage:
    Execute this script directly. It will automatically process the data, perform the preprocessing, apply
    stratified sampling, split the data, and then save the split datasets as CSV files in the designated
    nested directory structure.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import custom preprocessing utility from utils.data_util
from utils.data_util import Preprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define paths for annotated data input and output splits
ANNOT_PATH: Path = Path("./Dataset/Annotated Data/")
SPLIT_PATH: Path = Path("./Dataset/Clean Data/")


class Location(str, Enum):
    """Enumeration of supported locations for data processing."""

    AUSTRALIA = "au"
    UK = "uk"
    INDIA = "ind"


class Domain(str, Enum):
    """Enumeration of supported data domains."""

    GOOGLE = "Google"
    REDDIT = "Reddit"


class Task(str, Enum):
    """Enumeration of supported classification tasks."""

    SENTIMENT = "sentiment"
    SARCASM = "sarcasm"


class DataSplit(str, Enum):
    """Enumeration of dataset split types."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DataConfig(BaseModel):
    """
    Configuration for data processing parameters.

    Attributes:
        locations: List of locations to process.
        domains: List of data domains to process.
        tasks: List of classification tasks to process.
        train_size: Proportion of data to use for training.
        validation_size: Proportion of test data to use for validation.
        random_state: Seed for random operations to ensure reproducibility.
        discard_label: Label value to filter out from the dataset.
    """

    locations: List[Location] = Field(
        default=[Location.AUSTRALIA, Location.UK, Location.INDIA],
        description="Locations to process data for",
    )
    domains: List[Domain] = Field(
        default=[Domain.GOOGLE, Domain.REDDIT], description="Data domains to process"
    )
    tasks: List[Task] = Field(
        default=[Task.SENTIMENT, Task.SARCASM],
        description="Classification tasks to prepare data for",
    )
    train_size: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Proportion of data to use for training",
    )
    validation_size: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Proportion of test data to use for validation",
    )
    random_state: int = Field(default=7, description="Random seed for reproducibility")
    discard_label: int = Field(
        default=2, description="Label value to filter out from the dataset"
    )

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "DataConfig":
        """Validate that the split sizes make sense in combination."""
        train_size = self.train_size
        validation_size = self.validation_size

        # Calculate test size as a proportion of the full dataset
        test_size = (1.0 - train_size) * (1.0 - validation_size)

        # Check if any split would be too small
        min_size = 0.05  # At least 5% of data in each split
        if (
            train_size < min_size
            or test_size < min_size
            or validation_size * (1.0 - train_size) < min_size
        ):
            raise ValueError(
                f"Split sizes would result in too small partitions. "
                f"train={train_size}, val={validation_size * (1.0 - train_size)}, "
                f"test={test_size}"
            )

        return self

    model_config = {"extra": "forbid"}


class TaskDataset(BaseModel):
    """
    Dataset for a specific classification task.

    Attributes:
        data: The pandas DataFrame containing the dataset.
        task: The classification task this dataset is for.
        location: The geographic location this dataset represents.
        domain: The data domain this dataset belongs to.
    """

    data: pd.DataFrame
    task: Task
    location: Location
    domain: Domain

    model_config = {"arbitrary_types_allowed": True}

    def split(self, config: DataConfig) -> Dict[DataSplit, pd.DataFrame]:
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            config: Configuration parameters for the split.

        Returns:
            A dictionary mapping split types to DataFrames.
        """
        # Perform train-test split
        train, test = train_test_split(
            self.data,
            train_size=config.train_size,
            stratify=self.data[["label"]],
            random_state=config.random_state,
        )

        # Split test set into validation and final test sets
        val, test = train_test_split(
            test,
            train_size=config.validation_size,
            stratify=test[["label"]],
            random_state=config.random_state,
        )

        # Ensure no missing values in any split
        train.dropna(inplace=True)
        val.dropna(inplace=True)
        test.dropna(inplace=True)

        return {DataSplit.TRAIN: train, DataSplit.VALID: val, DataSplit.TEST: test}

    def save_splits(self, splits: Dict[DataSplit, pd.DataFrame]) -> None:
        """
        Saves the dataset splits to CSV files.

        Args:
            splits: Dictionary mapping split types to DataFrames.
        """
        # Create the directory structure if it doesn't exist
        output_dir = (
            SPLIT_PATH / self.task.title() / f"en-{self.location.upper()}" / self.domain
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_type, data in splits.items():
            output_file = output_dir / f"{split_type.value}.csv"
            data.to_csv(output_file, index=False, encoding="ascii", errors="ignore")
            logger.info(f"Saved {split_type.value} split to {output_file}")


def load_and_preprocess_data(
    location: Location, domain: Domain, config: DataConfig
) -> pd.DataFrame:
    """
    Loads and preprocesses data for a specific location and domain.

    Args:
        location: The location code (e.g., "au", "uk", "ind").
        domain: The data domain ("Google" or "Reddit").
        config: Configuration parameters.

    Returns:
        Preprocessed DataFrame.

    Raises:
        FileNotFoundError: If the Excel file doesn't exist.
        ValueError: If the specified domain sheet is not in the Excel file.
    """
    # Build the file path for the Excel file
    file_path = ANNOT_PATH / f"en-{location.upper()}.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    try:
        # Read the Excel sheet for the specified domain
        data = pd.read_excel(file_path, sheet_name=domain.value)

        # Preprocess the data
        preprocessor = Preprocess()
        data = preprocessor.preprocess(data)

        # Filter out reviews with the discard label
        data = data.loc[data["sentiment_label"] != config.discard_label].reset_index(
            drop=True
        )
        data.dropna(inplace=True)

        return data
    except ValueError as e:
        if f"'{domain.value}'" in str(e) and "is not in the" in str(e):
            raise ValueError(f"Sheet '{domain.value}' not found in {file_path}")
        raise


def prepare_task_dataset(
    data: pd.DataFrame, task: Task, location: Location, domain: Domain
) -> TaskDataset:
    """
    Prepares a dataset for a specific classification task.

    Args:
        data: The preprocessed DataFrame.
        task: The classification task.
        location: The data location.
        domain: The data domain.

    Returns:
        A TaskDataset object.
    """
    # Construct the task-specific label column name
    task_label_col = f"{task.value}_label"

    # Select and rename the necessary columns
    if task_label_col not in data.columns:
        raise ValueError(f"Column '{task_label_col}' not found in data")

    task_data = data[["id", "text", task_label_col]].copy()
    task_data.columns = ["id", "text", "label"]

    return TaskDataset(data=task_data, task=task, location=location, domain=domain)


def process_location_domain(
    location: Location, domain: Domain, config: DataConfig
) -> None:
    """
    Processes data for a specific location and domain.

    Args:
        location: The location to process.
        domain: The domain to process.
        config: Configuration parameters.
    """
    try:
        # Load and preprocess data
        logger.info(f"Processing {domain.value} data for {location.value}")
        data = load_and_preprocess_data(location, domain, config)

        # Process each classification task
        for task in config.tasks:
            logger.info(f"Preparing {task.value} task dataset")
            try:
                # Prepare task-specific dataset
                task_dataset = prepare_task_dataset(data, task, location, domain)

                # Split the dataset
                splits = task_dataset.split(config)

                # Save the splits
                task_dataset.save_splits(splits)

                # Log split sizes
                for split_type, split_data in splits.items():
                    logger.info(
                        f"{location.value}/{domain.value}/{task.value}/{split_type.value}: {len(split_data)} rows"
                    )

            except Exception as e:
                logger.error(f"Error processing {task.value} task: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing {location.value}/{domain.value}: {str(e)}")


def main() -> None:
    """
    Main function to process and split review data for multiple tasks.

    For each location and domain combination:
    1. Loads and preprocesses the data.
    2. For each classification task:
        a. Prepares a task-specific dataset.
        b. Splits the dataset into train/validation/test sets.
        c. Saves the splits to CSV files.
    """
    # Initialize configuration
    config = DataConfig()

    logger.info(f"Starting data processing with config: {config.model_dump()}")

    # Create output directory if it doesn't exist
    SPLIT_PATH.mkdir(parents=True, exist_ok=True)

    # Process each location-domain combination
    for location in tqdm(config.locations, desc="Processing locations"):
        for domain in config.domains:
            process_location_domain(location, domain, config)

    logger.info("Data processing completed successfully")


if __name__ == "__main__":
    main()
