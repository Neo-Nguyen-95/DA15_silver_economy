#%% LIB
import pandas as pd
import requests
from bs4 import BeautifulSoup
# import json

#%% SCRAPING
df_url = pd.read_csv('private/silver_economy_website.csv')
list_url = df_url['link'].values

def get_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    
    text = text.replace('\n', '')
    
    return text

scraped_data = {}
for index, url in enumerate(list_url):
    scraped_data[index] = get_text(url)

#%% EXPORT
# with open('data.json', 'w', encoding='utf-8') as file:
#     json.dump(scraped_data, file, indent=4, ensure_ascii=False)

df_export = pd.DataFrame({
    'data':scraped_data.values()
    })
df_export.to_csv('data.csv', encoding='utf-8-sig')
