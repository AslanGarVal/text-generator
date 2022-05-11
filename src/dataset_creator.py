# TODO: Import the data (stored in the data folder)
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from typing import List


class DatasetCreator:
    """
    This is the general class for reading a text dataset and generating a vocabulary from it.
    The import_data method is declared abstract to allow different implementations according to source.

    :param path_to_data_source: location from which data will be extracted
    """

    def __init__(self, path_to_data_source: str):
        self.path_to_data_source = path_to_data_source

    @abstractmethod
    def import_data(self) -> List[str]:
        """Imports data from path to source and preprocesses it into a list of sequences"""
        pass


class DatasetFromCsv(DatasetCreator):
    def import_data(self) -> List[str]:
        """
        Reads a csv file containing clickbait news headers and their date of publication,
        returns a list of all the headers, this will be used to build our dataset.

        :return: a list of strings containing the headers to be extracted
        """
        df = pd.read_csv(self.path_to_data_source)
        return df['headline_text'].values


