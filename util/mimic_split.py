import pandas as pd


# df = pd.read_csv('/root/autodl-tmp/data/mimic/sampled_master.csv')
df = pd.read_csv('/root/autodl-tmp/data/mimic/cleaned_master.csv')

df_train = df[df['split'] == 'train']
df_valid = df[df['split'] == 'valid']

# df_train.to_csv('sampled_train.csv', index=False)
# df_valid.to_csv('sampled_valid.csv', index=False)
df_train.to_csv('train.csv', index=False)
df_valid.to_csv('valid.csv', index=False)
