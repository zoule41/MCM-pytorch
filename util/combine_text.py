import pandas as pd

df = pd.read_csv('/root/autodl-tmp/cleaned_sampled_file.csv')

df['report_content'] = df['findings'] + " " + df['impression']
df['report_content'] = df['report_content'].apply(lambda x: x.strip())

df['complaint'] = df['indication'] + " " + df['history']
df['complaint'] = df['complaint'].apply(lambda x: x.strip())

output_path = '/root/autodl-tmp/new_cleaned_sampled_file.csv'

# 保存到新的CSV文件
df.to_csv(output_path, index=False)