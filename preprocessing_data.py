import pandas as pd
from transform_data import TransformData
from imblearn.under_sampling import RandomUnderSampler


class PreprocessingData:
    def __init__(self, filename):
        self.filename = filename

    def get_pickle_data(self):
        """
        Gets the dataframe from the pickle file

        :return: pickle data in dataframe
        """
        return pd.read_pickle(self.filename)

    def get_transformed_data(self, data):
        """
        Gets the one-hot vector data from the original data

        :param data: data in dataframe
        :return: dataframe that consists out of one-hot vectors
        """
        td = TransformData(data)
        return td.transform_data()

    def get_balanced_data(self, X, y):
        """
        Gets the undersampled data from X and y

        :param X: dataframe without labels
        :param y: series that contains the matching labels to X
        :return: undersampled data with separate matching labels
        """
        rus = RandomUnderSampler(random_state=0)
        return rus.fit_resample(X, y)

