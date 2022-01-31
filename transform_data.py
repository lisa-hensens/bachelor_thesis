import pandas as pd


class TransformData:
    def __init__(self, data):
        self.data = data

    def transform_data(self):
        """
        Transforms the column "hpo_all_name" of the original data to a symptom dataframe
        that has as columns all the different symptoms and a cell can have the value 1
        if a patient has a certain symptom or 0 if a patient does not have a certain symptom.
        :return: symptom dataframe with a separate labels dataframe
        """
        df = self.data

        # Make a list of all the possible symptoms (without duplicates)
        all_symptoms = []
        for symp_list in df["hpo_all_name"]:
            for symp in symp_list:
                if symp not in all_symptoms:
                    all_symptoms.append(symp)

        # Make empty dataframe with the symptoms as the column names
        df2 = pd.DataFrame(columns=all_symptoms)

        # If a patients has a symptom, it gets a 1 in the column of that symptom
        for i, symp_list in enumerate(df["hpo_all_name"]):
            for symp in symp_list:
                if symp in all_symptoms:
                    df2.at[i ,symp] = 1

        # Fill all the NaN values with 0 to match the No=0 and Yes=1 pattern in the data
        df2 = df2.fillna(0)
        return df2, df["label"]
