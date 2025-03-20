#%% LIB
import pandas as pd
from data import get_text, get_title

#%% SCRAPING




df_url = pd.read_csv('private/silver_economy_website.csv')

list_url = df_url['link'].values

scraped_data = {}

for index, url in enumerate(list_url):
    scraped_data[index] = get_title(url, level='h3')
    

df_export = pd.DataFrame({
    'data':scraped_data.values()
    })

df_export = df_url.join(df_export)
df_export.to_csv('data_article_temp.csv', encoding='utf-8-sig')
