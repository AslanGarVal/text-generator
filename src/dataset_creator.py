import functools
from abc import ABC, abstractmethod
import pandas as pd

from typing import List


class DatasetCreator(ABC):
    """
    This is the general class for reading a text dataset and generating a vocabulary from it.
    The import_data method is declared abstract to allow different implementations according to source.

    :param str path_to_data_source: location from which data will be extracted
    """

    def __init__(self, path_to_data_source: str, data: List[str] = None):
        self.path_to_data_source = path_to_data_source
        self.data = data

    @abstractmethod
    def import_data(self) -> List[str]:
        """Imports data from path to source and preprocesses it into a list of sequences"""
        pass

    def build_vocab(self) -> str:
        """Gets a list of sentences and returns all the distinct characters comprising every sentence

        :returns str String containing all distinct chars
        """
        distinct_chars_per_sentence = list(map(lambda s: set(s), self.data))
        return ''.join(functools.reduce(lambda x, y: x.union(y), distinct_chars_per_sentence))


class DatasetFromCsv(DatasetCreator):
    def import_data(self) -> None:
        """
        Reads a csv file containing clickbait news headers and their date of publication,
        returns a list of all the headers, this will be used to build our dataset.
        """
        df = pd.read_csv(self.path_to_data_source, quotechar='"').dropna()
        self.data = df['headline_text'].values
