import pandas

## 读取数据
df_1 = pandas.read_csv(r'./results_步法.csv')
df_1.columns = ['title','abstract',	'link']
df_1.insert(0, 'key_word', '步法')
df_2 = pandas.read_csv(r'./results_冬奥热点.csv')
df_2.columns = ['title','abstract',	'link']
df_2.insert(0, 'key_word', '冬奥热点')
df_3 = pandas.read_csv(r'./results_冬奥瞬间.csv')
df_3.columns = ['title','abstract',	'link']
df_3.insert(0, 'key_word', '冬奥瞬间')
df_4 = pandas.read_csv(r'./results_跳跃.csv')
df_4.columns = ['title','abstract',	'link']
df_4.insert(0, 'key_word', '跳跃')
#print(df_1)
print([df_1.shape, df_2.shape, df_3.shape, df_4.shape])

#缺省值 这里由于是通过爬虫获取的数据并且在数据量不大的情况下，所以直接使用剔除缺省数据的方式处理
df_1 = df_1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_2 = df_2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_3 = df_3.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_4 = df_4.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print([df_1.shape, df_2.shape, df_3.shape, df_4.shape])

#重复值  重复项仅保留一条记录
df_1 = df_1.drop_duplicates(keep=False)
df_2 = df_2.drop_duplicates(keep=False)
df_3 = df_3.drop_duplicates(keep=False)
df_4 = df_4.drop_duplicates(keep=False)
print([df_1.shape, df_2.shape, df_3.shape, df_4.shape])

df_1.to_csv(r'./preprocess_步法.csv', header=True, index=False, encoding='utf-8')
df_2.to_csv(r'./preprocess_冬奥热点.csv', header=True, index=False, encoding='utf-8')
df_3.to_csv(r'./preprocess_东奥瞬间.csv', header=True, index=False, encoding='utf-8')
df_4.to_csv(r'./preprocess_跳跃.csv', header=True, index=False, encoding='utf-8')
##这里将多个处理结果合并
df = [df_1, df_2, df_3, df_4]
df = pandas.concat(df)
df.to_csv(r'./preprocess_all.csv', header=True, index=False, encoding='utf-8')
#print(df)
