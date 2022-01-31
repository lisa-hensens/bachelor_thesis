import itertools
import numpy as np


class DataAnalysis:
    def __init__(self, data):
        """
        :param data: dataframe
        """
        self.data = data

    def remove_duplicates(self):
        """
        Removes the duplicates from each element, which are lists, in the column 'hpo_all_name' of the data
        """
        no_duplicates = [list(set(patient)) for patient in self.data['hpo_all_name']]
        self.data['hpo_all_name'] = no_duplicates

    def get_counts(self, label, frequent):
        """
        Returns the combinations of symptoms with their number of occurrences for the given genetic disorder

        :param frequent:
        :param label: the genetic disorder for which you want the symptom combinations
        :return: lists that contain top 5 of most frequent symptoms by index, counts and the
                 corresponding combinations in names of symptoms
        """
        filter_data = self.data[self.data['label'] == label]
        symptoms = filter_data['hpo_all_name']
        # Get the number of symptoms per patient
        lensym = [len(list(set(x))) for x in symptoms]
        # Get the patient with the least amount of symptoms for their diagnosis
        patient_least = list(set(symptoms.iloc[np.argmin(lensym)]))
        # Combining the patient with the least amount of symptoms with the 5 most frequent symptoms
        for symp in frequent:
            if symp not in patient_least:
                patient_least.append(symp)
        # Get all combinations
        all_combinations = []
        for i in range(2, len(patient_least)+1): # i represents the length of the combination
            combs = itertools.combinations(patient_least, i)
            combs_list = [list(x) for x in list(combs)]
            all_combinations += combs_list
        # Get counts per combination
        cnt = {all_combinations.index(x):0 for x in all_combinations}
        for patient in symptoms:
            for comb in all_combinations:
                if set(comb).issubset(set(patient)):
                    cnt[all_combinations.index(comb)] += 1
        # Sort list to get most frequent combinations
        sorted_key = sorted(cnt, key=cnt.get, reverse=True)[:5]
        sorted_count = [cnt.get(i) for i in sorted_key]
        sorted_combs = [all_combinations[i] for i in sorted_key]
        return sorted_key, sorted_count, sorted_combs

    def get_frequent_symptoms(self, label):
        """
        Returns the top 5 most frequent symptoms for the given genetic disorder

        :param label: the genetic disorder for which you want the symptom combinations
        :return: lists that contain top 5 of most frequent symptoms by index, counts and the
                 corresponding combinations in names of symptoms
        """
        filter_data = self.data[self.data['label'] == label]
        symptoms = filter_data['hpo_all_name']
        # Get all symptoms that occur for the given genetic disorder based on the dataset
        all_symptoms = []
        for symp_list in symptoms:
            for symp in symp_list:
                if symp not in all_symptoms:
                    all_symptoms.append(symp)
        # Get counts per symptom
        cnt = {all_symptoms.index(x):0 for x in all_symptoms}
        for patient in symptoms:
                for symp in list(set(patient)):
                    cnt[all_symptoms.index(symp)] += 1
        # Sort list to get most frequent combinations
        sorted_key = sorted(cnt, key=cnt.get, reverse=True)[:5]
        sorted_count = [cnt.get(i) for i in sorted_key]
        sorted_symps = [all_symptoms[i] for i in sorted_key]
        return sorted_key, sorted_count, sorted_symps

    def labels_frequent_symptoms(self, frequent):
        """
        This function gives a list of genetic disorders that each have the 5 frequent symptoms occur
        at least once in their patients.
        :return: labels lists with an accomponying frequency counts list
        """
        labels = []

        for symp in range(len(frequent)):
            disorders = []
            for i in range(len(self.data)):
                label = self.data['label'][i]
                if label not in disorders:
                    symptoms = self.data['hpo_all_name'][i]
                    if frequent[symp] in symptoms:
                        disorders.append(label)
            labels.append(disorders)
        counts = [len(x) for x in labels]
        return labels, counts

    def filter_basic_symptoms(self):
        """
        Filters out all the non-differenciating symptoms to get a more distinct profile of
        each genetic disorder. It also filters out all the times both Intellectual Disability
        and a version with severity appear in symptoms list of a patient. Then only the
        symptom with severity will remain.
        """
        symptoms = self.data['hpo_all_name']
        frequent_symptoms = ['Delayed speech and language development', 'Motor delay']
        for patient in symptoms.index.values:
            for symp in frequent_symptoms:
                if symp in symptoms[patient]:
                    symptoms[patient].remove(symp)
        self.data['hpo_all_name'] = symptoms

    def filter_severity(self):
        """
        Filters out the symptom Intellectual Disability if the symptom Intellectual disability is
        mentioned again with a grade of severity such as mild, moderate or severe.
        """
        symptoms = self.data['hpo_all_name']
        for patient in symptoms.index.values:
            if "Intellectual disability" in symptoms[patient]:
                if "Intellectual disability, mild" in symptoms[patient]:
                    symptoms[patient].remove("Intellectual disability")
                elif "Intellectual disability, moderate" in symptoms[patient]:
                    symptoms[patient].remove("Intellectual disability")
                elif "Intellectual disability, severe" in symptoms[patient]:
                    symptoms[patient].remove("Intellectual disability")
        self.data['hpo_all_name'] = symptoms

