import csv
import os

import pandas as pd
from sklearn.model_selection import train_test_split


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


class HallmarksDataset:
    def __init__(self,
                 in_text_path: str,
                 in_label_path: str,
                 out_train_path: str,
                 out_test_path: str):
        self.in_text_path = in_text_path
        self.in_label_path = in_label_path
        self.out_train_path = out_train_path
        self.out_test_path = out_test_path

    def load(self):
        df = self._parse(self.in_text_path)
        label_df = self._parse(self.in_label_path)
        df['label'] = label_df.document.map(self._find_labels)
        df = df[df.label.map(lambda x: len(x) > 0)]

        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        self._to_txt(train_df, self.out_train_path)
        self._to_txt(test_df, self.out_test_path)
        return train_df, test_df

    def _parse(self, base_dir):
        strip = lambda x: x.replace('\n', ' ').replace('\r', ' ')
        rows = []

        for file in os.listdir(base_dir):
            reader = open(os.path.join(base_dir, file), mode='r')

            document_id = file.split('.txt')[0]
            text = reader.read()
            document = strip(text)
            reader.close()
            rows.append({'id': document_id, 'document': document})

        return pd.DataFrame(rows)

    def _find_labels(self, text):
        categories = [
            'Sustaining proliferative signaling',
            'Evading growth suppressors',
            'Resisting cell death',
            'Enabling replicative immortality',
            'Inducing angiogenesis',
            'Activating invasion and metastasis',
            'Genomic instability and mutation',
            'Tumor promoting inflammation',
            'Cellular energetics',
            'Avoiding immune destruction',
        ]

        return [category for category in categories if category in text]

    def _to_txt(self, df, output_file):
        df[['document']].to_csv(output_file,
                                sep='\t',
                                index=False,
                                header=False,
                                quoting=csv.QUOTE_NONE,
                                encoding='utf-8')
