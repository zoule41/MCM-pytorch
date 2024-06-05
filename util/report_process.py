import pandas as pd
import re


def clean_report_mimic_cxr(report):
    # report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('  ', ' ') \
    #     .replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
    #     .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
    #     .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
    #     .strip().lower().split('. ')
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('  ', ' ').replace('..', '.') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    return tokens


df = pd.read_csv('master.csv')
df['cleaned_report'] = df['report_content'].apply(clean_report_mimic_cxr)
df['cleaned_complaint'] = df['complaint'].apply(clean_report_mimic_cxr)


df['cleaned_report'] = df['cleaned_report'].apply(lambda tokens: ' . '.join(tokens) + ' .')
df['cleaned_complaint'] = df['cleaned_complaint'].apply(lambda tokens: ' . '.join(tokens) + ' .')


df.to_csv('cleaned_master.csv', index=False)
