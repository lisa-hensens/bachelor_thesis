class RuleBased:
    def __init__(self, labels):
        """
        :param labels: list of all possible labels that occur throughout the data
        """
        self.labels = labels

    def get_label(self, patient):
        """
        Gets predicted label based on the input patient data with default being undiagnosed.
        :param patient: list with symptoms
        :return: diagnosis as a string
        """
        if set(['Highly arched eyebrow', 'Prolonged neonatal jaundice', 'Microcephaly', 'Smooth philtrum', 'Pointed chin', 'Primary microcephaly']).issubset(set(patient)):
            return 'spop_1'
        elif set(['Cleft palate', 'Velopharyngeal insufficiency’']).issubset(set(patient)) or set(["Generalized hypotonia", 'Velopharyngeal insufficiency']).issubset(set(patient)):
            return '22q11'
        elif set(['Autism', 'Intellectual disability, severe']).issubset(set(patient)):
            return 'adnp'
        elif set(['Abnormality of the face', 'Generalized hypotonia']).issubset(set(patient)) or set(['Hypoplasia of the corpus callosum', 'Generalized hypotonia']).issubset(set(patient)):
            return 'cltc'
        elif set(['Generalized hypotonia', 'Abnormality of movement']).issubset(set(patient)) or set(['Generalized hypotonia', 'Behavioral abnormality']).issubset(set(patient)) or set(['Generalized hypotonia', 'Intellectual disability, severe']).issubset(set(patient)):
            return 'ddx3x'
        elif set(['Microcephaly', 'Febrile seizure (within the age range of 3 months to 6 years)']).issubset(set(patient)):
            return 'dyrk1a'
        elif set(['Generalized hypotonia', 'Bulbous nose']).issubset(set(patient)) or set(['Generalized hypotonia', 'Overfriendliness']).issubset(set(patient)):
            return 'kansl1'
        elif set(['Wide mouth', 'Prominent nasal tip']).issubset(set(patient)) or set(['Pointed chin', 'Prominent nasal tip']).issubset(set(patient)):
            return 'kdm3b'
        elif set(['Intellectual disability', 'Abnormality of the face']).issubset(set(patient)):
            return 'pacs1'
        elif set(['Intellectual disability, severe', 'Feeding difficulties']).issubset(set(patient)):
            return 'pura'
        elif set(['Hypothyroidism', 'Large forehead']).issubset(set(patient)) or set(['Prominent nasal bridge', 'Large forehead']).issubset(set(patient)) or set(['Prominent nasal bridge', 'Hypertolerism']).issubset(set(patient)) or set(['Prominent nasal bridge', 'Prominent nasal tip']).issubset(set(patient)) or set(['Prominent nasal bridge', 'Sparse and thin eyebrow']).issubset(set(patient)):
            return 'spop_2'
        elif set(['Neurodevelopment delay', 'Hypotonia']).issubset(set(patient)):
            return 'wac'
        elif set(['Broad forehead', 'Malar flattening']).issubset(set(patient)) or set(['Broad forehead', 'Thick lower lip vermilion']).issubset(set(patient)):
            return 'yy1'
        elif set(['Short stature', 'Birth length greater than 97th percentile']).issubset(set(patient)):
            if 'Macrodontia of permanent maxillary central incisor' or 'Autism' or 'Bulbous nose' in patient:
                return 'ankrd'
            elif 'Vomiting' or 'Small hand' in patient:
                return 'ppm1d'
            elif 'Intellectual disability, severe' or 'Downslanted palpebral fissures' in patient:
                return 'son'
        else:
            return 'Undiagnosed'