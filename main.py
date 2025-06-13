#%% LIB
from data import DataRepository
import pandas as pd
pd.set_option('display.max_columns', None)
# import seaborn as sns
# import matplotlib.pyplot as plt
# from bertopic import BERTopic

#%% EDA
repo = DataRepository()

# repo.grouping_lda()

# df_full = repo.load_doc_sentiment()
# df_full.head()

#%% TRANSCRIBE VIDEO
topic_name = "fb audio"

repo.download_audio_yt(url='https://www.facebook.com/yeah1tv/videos/522671014216588/', 
                       file_name=f'data_output/{topic_name}')

text = repo.transcribe_audio(f'data_output/{topic_name}.m4a')

with open(f'data_output/{topic_name}.txt', 'w') as file:
    file.write(text)


#%% SENTIMENT ÂNLYSIS
# df_sen = (df_full.groupby(['topics', 'sentiment']).agg('count')
#           .reset_index()
#           .sort_values(['topics', 'sentiment'], ascending=False)
#           .drop([6,7,8])
#           )

# plt.figure(dpi=200)
# sns.barplot(data=df_sen, x='doc', y='topics', hue='sentiment', 
#             palette=['#5F8B4C', '#FFDDAB', '#FF9A9A'],
#             edgecolor='black', linewidth=1
#             )
# plt.xlabel('Số lượng bài viết')
# plt.ylabel('Chủ đề chung')
# plt.show()

# df_full.drop_duplicates(subset=['topics', 'sentiment'], keep='first').to_csv('sample.csv', encoding='utf-8-sig')

#%% GROUPING WITH BERTOPIC => FAIL
# docs = df_full[df_full['topics']!='Khác']['doc'].to_list()


# # repo.get_batch_openai_embedding(documents=docs)
# docs_embedded = repo.load_embedding('embedding.npy')
# docs_filter = repo.remove_long_docs(docs)

# topic_model = BERTopic(min_topic_size=10)

# topics, prob = topic_model.fit_transform(docs_filter, docs_embedded)

# print(topic_model.get_topic_info())

# topic_model.visualize_barchart(top_n_topics=10)
