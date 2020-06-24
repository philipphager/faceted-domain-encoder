import csv
import os

import pandas as pd


class OhsumedDataset:
    def __init__(self,
                 in_path: str,
                 out_path: str):
        self.in_path = in_path
        self.out_path = out_path

    def load(self):
        df = self._parse(self.in_path)
        self._to_txt(df, self.out_path)
        return df

    def _parse(self, base_dir):
        strip = lambda x: x.replace('\n', ' ').replace('\r', ' ')
        rows = []

        for directory in os.listdir(base_dir):
            category = self._get_category(directory)

            if directory == '.DS_Store':
                continue

            for file in os.listdir(os.path.join(base_dir, directory)):
                reader = open(os.path.join(base_dir, directory, file), mode='r')

                document_id = file
                text = reader.read()
                document = strip(text)
                title = text.split('\n', 1)[0]
                reader.close()
                rows.append({'id': document_id, 'label': category, 'title': title, 'document': document})

        df = pd.DataFrame(rows)
        return df.groupby(['id', 'title', 'document']) \
            .label.apply(list) \
            .reset_index()

    def _to_txt(self, df, output_file):
        df[['document']].to_csv(output_file, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE,
                                encoding='utf-8')

    def _get_category(self, directory):
        directory2category = {
            'C01': 'Bacterial Infections and Mycoses',
            'C02': 'Virus Diseases',
            'C03': 'Parasitic Diseases',
            'C04': 'Neoplasms',
            'C05': 'Musculoskeletal Diseases',
            'C06': 'Digestive System Diseases',
            'C07': 'Stomatognathic Diseases',
            'C08': 'Respiratory Tract Diseases',
            'C09': 'Otorhinolaryngologic Diseases',
            'C10': 'Nervous System Diseases',
            'C11': 'Eye Diseases',
            'C12': 'Urologic and Male Genital Diseases',
            'C13': 'Female Genital Diseases and Pregnancy Complications',
            'C14': 'Cardiovascular Diseases',
            'C15': 'Hemic and Lymphatic Diseases',
            'C16': 'Neonatal Diseases and Abnormalities',
            'C17': 'Skin and Connective Tissue Diseases',
            'C18': 'Nutritional and Metabolic Diseases',
            'C19': 'Endocrine Diseases',
            'C20': 'Immunologic Diseases',
            'C21': 'Disorders of Environmental Origin',
            'C22': 'Animal Diseases',
            'C23': 'Pathological Conditions, Signs and Symptoms'
        }
        return directory2category[directory]
