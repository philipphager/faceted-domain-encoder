import re

import pandas as pd

from experiments.ablation.util.mail import EmailReplyParser


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
                content = ' '.join([line for line in content.split('\n') if not self._drop_line(line)])
                content = content.replace('\r', ' ').replace('\n', '').replace('\t', ' ')
                content = re.sub('\s+', ' ', content)
                content = content.split('Regards')[0].split('regards')[0]

                if len(content) > 0 and not content.isspace() and not self._drop_message(content):
                    messages.append(content)

        return messages

    def _strip(self, text):
        text = str(text.decode('UTF-8'))
        return text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

    def _drop_line(self, line):
        # List of lines that can be removed that we identified manually, might be incomplete
        prefixes = [
            'Cc:',
            'This e-mail (including attachments) contains contents owned by Rolls-Royce',
            'This e-mail and any files',
            'Neither the company nor any subsidiar',
            'The recipient should check this email',
            'The data contained in, or attached to, this e-mail',
            '____',
            '----'
        ]

        return any([line.strip().startswith(p) for p in prefixes])

    def _drop_message(self, text):
        # List of auto.reply fragments that we identified manually, might be incomplete
        messages = [
            'Auto forwarded by a Rule',
            '** Security Notice',
            '(c) 2012 Rolls-Royce plc Registered office',
            'An e-mail response to this address may be subject to interception or monitoring',
            'This e-mail and any attachments may contain confidential or privileged information',
            'Confidential Communication',
            'confidential',
        ]

        return any([m in text for m in messages])

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
