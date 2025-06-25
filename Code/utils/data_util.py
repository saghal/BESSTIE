import os
import re
import sys
from pathlib import PurePath

import emoji
import ftfy
import pandas as pd


class InvalidModelError(Exception):
    """
    Error for model issues.
    """

    pass


class InvalidInputError(Exception):
    """
    Error for invalid input.
    """

    pass


class Preprocess:
    """
    Preprocess text data by cleaning text and converting ratings to binary.
    """

    def __init__(self) -> None:
        """
        Initialize the Preprocess object.
        """
        pass

    def __remove_emojis(self, text: str) -> str:
        """
        Remove emojis from the text.

        Args:
            text (str): The text to remove emojis from.

        Returns:
            str: The text with all emojis removed.

        Raises:
            InvalidInputError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected string, got {type(text)}")
        return emoji.replace_emoji(text, replace="")

    def __remove_hyper(self, text: str) -> str:
        """
        Remove URLs from the text.

        Args:
            text (str): The text to remove URLs from.

        Returns:
            str: The text with all URLs removed.

        Raises:
            InvalidInputError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected string, got {type(text)}")
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def __fix_contractions(self, text: str) -> str:
        """
        Fix spacing errors in contractions (e.g., converting 'aren 't' to 'aren't').

        Args:
            text (str): The text with spacing errors in contractions.

        Returns:
            str: The text with contractions properly formatted.
        """
        return re.sub(r"(\w+)\s+'(\w+)", r"\1'\2", text)

    def __remove_unbalanced(self, text: str, open_char: str, close_char: str) -> str:
        """
        Remove unbalanced occurrences of the specified characters from the text.

        Args:
            text (str): The text to process.
            open_char (str): The opening character (e.g., '[').
            close_char (str): The closing character (e.g., ']').

        Returns:
            str: The text with unbalanced characters removed.
        """
        indices_to_remove = set()
        stack = []
        for i, ch in enumerate(text):
            if ch == open_char:
                stack.append(i)
            elif ch == close_char:
                if stack:
                    stack.pop()
                else:
                    indices_to_remove.add(i)
        indices_to_remove.update(stack)
        return "".join(ch for i, ch in enumerate(text) if i not in indices_to_remove)

    def __clean(self, text: str) -> str:
        """
        Clean the text by removing URLs, emojis, fixing contractions, and eliminating unbalanced characters.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.

        Raises:
            InvalidInputError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected string, got {type(text)}")
        text = self.__remove_hyper(text)
        text = self.__remove_emojis(text)
        text = ftfy.fix_text(text)
        text = self.__fix_contractions(text)
        text = re.sub(r"!?\[gif\]\([^)]*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\(", r"\1", text)
        for open_char, close_char in [("[", "]"), ("(", ")"), ("{", "}")]:
            text = self.__remove_unbalanced(text, open_char, close_char)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    def preprocess(
        self, df: pd.DataFrame, text: str = "review", label: str = "rating"
    ) -> pd.DataFrame:
        """
        Preprocess the DataFrame by converting ratings to binary and cleaning the text column.

        Args:
            df (pd.DataFrame): The DataFrame containing the text and rating columns.
            text (str, optional): The name of the text column. Defaults to "review".
            label (str, optional): The name of the rating column. Defaults to "rating".

        Returns:
            pd.DataFrame: The preprocessed DataFrame with binary ratings and cleaned text.
        """
        df[label] = df[label].apply(lambda x: 1 if x > 2 else 0)
        df.dropna(inplace=True)
        df[text] = df[text].apply(lambda x: self.__clean(x))
        return df


def get_data(path: str) -> pd.DataFrame:
    """
    Load data from a file into a DataFrame based on the file extension.

    Supports CSV, TSV, JSON, and JSONL formats. Exits the program if the file does
    not exist or if the file type is unsupported.

    Args:
        path (str): The file path to load data from.

    Returns:
        pd.DataFrame: The loaded DataFrame with missing values dropped.

    Raises:
        SystemExit: If the file does not exist or the file type is unsupported.
    """
    if not os.path.exists(path):
        print(f"Error: The data path '{path}' does not exist.")
        sys.exit(-1)

    path = os.path.join(*(list(PurePath(path).parts)))
    file_type = path.split(".")[-1]

    if file_type in ["csv", "tsv"]:
        sep = "\t" if file_type == "tsv" else ","
        data = pd.read_csv(
            path, sep=sep, encoding="ascii", encoding_errors="ignore"
        ).dropna()
    elif "json" in file_type:
        data = pd.read_json(
            path, lines="l" in file_type, encoding="ascii", encoding_errors="ignore"
        ).dropna()
    else:
        print(f"Error: The file type '{file_type}' is not supported.")
        print("Please provide CSV, TSV, JSON, or JSONL.")
        sys.exit(-1)

    return data


def post_process(text: str) -> str:
    """
    Post-process the text by normalizing, removing special characters, and limiting the output.

    The process includes converting to lowercase, stripping spaces, removing unwanted
    characters and substrings, and limiting the output to a maximum of three alphabetic words.

    Args:
        text (str): The text to post-process.

    Returns:
        str: The cleaned and normalized text.
    """
    word = text.lower().strip()
    if "[/inst]" in word:
        word = word.split("[/inst]")[1]
    for char in [
        "/",
        "\\",
        "*",
        "?",
        "[",
        "]",
        "{",
        "}",
        "(",
        ")",
        ",",
        ":",
        '"',
        "'",
        ".",
        "`",
    ]:
        word = word.replace(char, "")
    word = word.strip()
    if "<end_of_turn>model" in word:
        word = word.split("<end_of_turn>model")[1]
    word = word.split("<end_of_turn>")[0]
    word = word.split("<s>")[0].split("</s>")[0].strip().split("\n")[0]
    word = word.replace(">", "").replace("<", "")
    word = word.split("\n")[0]
    word = word.split(" or ")[0]
    word = word.split(" and ")[0]
    if len(word.split()) > 3:
        word = " ".join(word.split()[:3])
    word = word.strip()
    words = [i for i in word.split(" ") if i.isalpha()]
    return " ".join(words)
