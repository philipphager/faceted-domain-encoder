import re

import pandas as pd
from cleantext import clean
from experiments.ablation.util.mail import EmailReplyParser

GERMAN_PHONE = re.compile('(\(?([\d \-\)\–\+\/\(]+)\)?([ .\-–\/]?)([\d]+))')
IMAGE_ATTACHMENT = re.compile('[a-zA-Z]*image[a-zA-Z]*')
MAILTO = re.compile('[a-zA-Z]*<mailto:>[a-zA-Z]*')
MAIL = re.compile('[a-zA-Z]*@[a-zA-Z]*\.[a-zA-Z]*')
SPACE = re.compile('\s+')
PUNCTUATION = '!"$%&\'()*+,.:;<=>?@[\\]^_`{|}~'


class EmailDataset:
    def __init__(self,
                 in_path,
                 out_train_path,
                 out_test_path,
                 split=0.5):
        self.in_path = in_path
        self.out_train_path = out_train_path
        self.out_test_path = out_test_path
        self.split = split

    def load(self):
        df = pd.read_pickle(self.in_path)

        # Extract latest email per conversation with entire email chain
        df = df.sort_values('delivery_time', ascending=False)
        df = df.groupby('conversation_topic').head(1)

        # Split threads into messages
        df.plain_text_body = df.plain_text_body.map(lambda x: x.decode('UTF-8'))
        df['messages'] = df.plain_text_body.map(self._split_thread)
        df = self._unpack_messages(df)

        df['length'] = df.text.map(len)
        min_document_length = df.text.map(len).quantile(0.1)

        df = df[df.text.map(len) > min_document_length]
        df = df.drop_duplicates()

        # Keep threads together during split
        train_df = df.head(int(len(df) * self.split))
        test_df = df.tail(int(len(df) * self.split) + 1)

        train_df.text.to_csv(self.out_train_path, index=False, header=False)
        test_df.text.to_csv(self.out_test_path, index=False, header=False)
        return train_df, test_df

    def _split_thread(self, text):
        message = EmailReplyParser.read(text)
        messages = []

        for fragment in message.fragments:
            if not fragment.headers:
                content = fragment._content
                content = self._clean(content)

                if not content.isspace() and not self._drop_message(content):
                    messages.append(content)

        return messages

    def _clean(self, text):
        text = clean(text,
                     no_line_breaks=True,
                     no_urls=True,
                     no_emails=True,
                     no_phone_numbers=True,
                     replace_with_url='',
                     replace_with_email='',
                     replace_with_phone_number='')
        text = GERMAN_PHONE.sub('', text)
        text = IMAGE_ATTACHMENT.sub('', text)
        text = MAIL.sub('', text)
        text = MAILTO.sub('', text)
        text = text.translate(str.maketrans(PUNCTUATION, ' ' * len(PUNCTUATION)))
        text = SPACE.sub(' ', text)
        return text

    def _drop_message(self, text):
        messages = [
            'cc',
            'auto forwarded',
            'security notice',
            '(c) 2012 rolls-royce plc registered office',
            'subject to interception or monitoring',
            'confidential',
        ]

        return any([m in text.lower() for m in messages])

    def _unpack_messages(self, frame):
        topics = []
        messages = []

        for i, row in frame.iterrows():
            topics.extend([row.conversation_topic] * len(row.messages))
            messages.extend(row.messages)

        frame = pd.DataFrame({'topic': topics, 'text': messages})
        frame = frame.groupby('text').head(1)
        frame = frame.reset_index()
        return frame
