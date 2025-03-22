#%% LIB
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import certifi

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

#%% FOR SCRAPING
def get_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(
        url, 
        headers=headers, 
        # verify=False
        )
    soup = BeautifulSoup(response.text, "html.parser")
    text_array = soup.find_all(["p"])
    
    text_array = [text_chunk.text for text_chunk in text_array]
    
    result = " ".join(text for text in text_array)
    
    return result

def get_title(url, level='h1'):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(
        url, 
        headers=headers, 
        # verify=False
        )
    soup = BeautifulSoup(response.text, "html.parser")
    text_array = soup.find_all([level])
    
    result = [text_chunk.text.strip() for text_chunk in text_array]
    
    return result

#%% GET LINK & TITLE FROM PUBLISHER
def get_article_link(class_name, url_page):
    
    response = requests.get(url_page)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    content_array = soup.find_all("a", class_=class_name)
    
    data_storage = {'title': [],
                    'link': []}
    
    for content in content_array:
        title = content.text.strip().lower()
        link = content.get('href')
        if ('nct' in title or 
            'cao tuổi' in title or 
            'già' in title or
            'cụ' in title
            ):        
            data_storage['title'].append(title)
            data_storage['link'].append(link)
        
    df = pd.DataFrame(data_storage)
        
    return df
    

#%% FOR NGAY MOI ONLINE 
num_range = range(0, 1021, 15)

df_article_ngaymoionline = pd.DataFrame()

for num in num_range:
    url_page_temp = f"https://ngaymoionline.com.vn/vi-nguoi-cao-tuoi&s_cond=&BRSR={num}"
    
    df_temp = get_article_link(class_name='article-link f0', url_page=url_page_temp)
    
    df_article_ngaymoionline = pd.concat([df_article_ngaymoionline, df_temp], axis='rows')
    
df_article_ngaymoionline = df_article_ngaymoionline.drop_duplicates()

df_article_ngaymoionline.to_csv('ngaymoionline_article-link.csv', index=False, encoding='utf-8-sig')

#%% FOR DAN TRI
num_range = range(1, 31)

df_article_dantri = pd.DataFrame()

for num in num_range:
    url_page_temp = f"https://dantri.com.vn/tim-kiem/ng%C6%B0%E1%BB%9Di+cao+tu%E1%BB%95i.htm?pi={num}"
    
    df_temp = get_article_link(class_name='dt-text-black-mine', url_page=url_page_temp)
    
    df_article_dantri = pd.concat([df_article_dantri, df_temp], axis='rows')
    
df_article_dantri = df_article_dantri.drop_duplicates()
    
df_article_dantri.to_csv('dantri_article-link.csv', index=False, encoding='utf-8-sig')



