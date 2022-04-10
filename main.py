import pandas as pd
import numpy as np
from apyori import apriori
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import re

df_data = pd.read_csv('datasets/winemag-data-130k-v2.csv') # 读取数据
df_data = df_data.drop(['Unnamed: 0', 'title', 'taster_twitter_handle'], axis=1)

# 用region_1的值来填充region_2的缺失值
df_data['region_2'].fillna(value=df_data['region_1'], inplace=True)


# 使用最高频率值来填补缺失值
simpleImp = SimpleImputer(strategy="most_frequent")
data_columns = df_data.columns
df_nona = pd.DataFrame(simpleImp.fit_transform(df_data))
df_nona.columns = data_columns

# for index, value in df_nona['points'].items():
#     if 80 <= value < 84:
#         df_nona['points'].at[index] = '80-84pts'
#     elif 84 <= value < 88:
#         df_nona['points'].at[index] = '84-88pts'
#     elif 88 <= value < 92:
#         df_nona['points'].at[index] = '88-92pts'
#     elif 92 <= value < 96:
#         df_nona['points'].at[index] = '92-96pts'
#     elif 96 <= value:
#         df_nona['points'].at[index] = '96-100pts'
#
# for index, value in df_nona['price'].items():
#     if 0 <= value < 10:
#         df_nona['price'].at[index] = '0-10$'
#     elif 10 <= value < 20:
#         df_nona['price'].at[index] = '10-20$'
#     elif 20 <= value < 30:
#         df_nona['price'].at[index] = '20-30$'
#     elif 30 <= value < 40:
#         df_nona['price'].at[index] = '30-40$'
#     elif 40 <= value < 50:
#         df_nona['price'].at[index] = '40-50$'
#     elif 50 <= value < 60:
#         df_nona['price'].at[index] = '50-60$'
#     elif 60 <= value < 70:
#         df_nona['price'].at[index] = '60-70$'
#     elif 70 <= value < 80:
#         df_nona['price'].at[index] = '70-80$'
#     elif 80 <= value < 90:
#         df_nona['price'].at[index] = '80-90$'
#     elif 90 <= value < 100:
#         df_nona['price'].at[index] = '90-100$'
#     elif 100 <= value < 150:
#         df_nona['price'].at[index] = '100-150$'
#     elif 150 <= value < 200:
#         df_nona['price'].at[index] = '150-200$'
#     elif 200 <= value < 500:
#         df_nona['price'].at[index] = '200-500$'
#     elif 500 <= value < 1000:
#         df_nona['price'].at[index] = '500-1000$'
#     elif 1000 <= value < 2000:
#         df_nona['price'].at[index] = '1000-2000$'
#     elif 2000 <= value:
#         df_nona['price'].at[index] = '2000-$'

# data = df_nona.to_dict('split')['data']
# stop_words = ['is', 'and', 'the', 'a', 'an', 'this', 'description:', 'to', 'with', 'of', 'in', 'wine', 'on', 'that', 'by', 'are', 'from', 'it\'s', 'it',
#               'Is', 'And', 'The', 'A', 'An', 'This', 'Description:', 'To', 'With', 'Of', 'In', 'Wine', 'On', 'That', 'By', 'Are', 'From', 'It\'s', 'It',
#               'I']
# for row in data:
#     for i in range(len(data_columns)):
#         if i != 1:
#             row[i] = data_columns[i] + ': ' + row[i]
#     description = row.pop(1)
#     words = re.split(r"\b[\.,\s\n\r\n]+?\b", description)
#
#     for word in words:
#         if word not in stop_words:
#             row.append('word: '+word)
#
# for i in range(5):
#     print(data[i])

# res = apriori(transactions=data, min_confidence=0.6)
# for rule in res:
#     print(str(rule))
#
# num_Roger = df_nona['taster_name'].value_counts().at['Roger Voss']
# num_France = df_nona['country'].value_counts().at['France']
# num_total = df_nona.shape[0]
#
# expect_num = int(num_France*num_Roger/num_total)
# observed_num = int(0.14516315178001246*num_total)
#
# chi_square = (expect_num - observed_num)*(expect_num - observed_num) / expect_num

# fig, ax = plt.subplots()
# df_France_taster = df_nona.loc[df_nona['country'] == 'France', ['taster_name']]
# df_France_taster['taster_name'].hist(xrot=30, ax=ax, bins=10)

fig, ax = plt.subplots()
df_US_Roger_prov = df_nona.loc[df_nona['country'] == 'US'].loc[df_nona['taster_name'] == 'Roger Voss', ['province']]
df_US_Roger_prov['province'].hist(xrot=60, ax=ax, bins=40)

plt.show()