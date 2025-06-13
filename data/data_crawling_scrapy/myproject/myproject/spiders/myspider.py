#%% LIB
import scrapy
import pandas as pd

#%% LOAD LINKS
ngaymoionline_links = pd.read_csv('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/resource_links/ngaymoionline_article-link.csv')

dantri_links = pd.read_csv('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/resource_links/dantri_article-link.csv')

manual_links = pd.read_csv('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA15_silver_economy/data/resource_links/manual-collected_article-link.csv')
manual_links = manual_links[manual_links['lang'] == 'vie']


#%% CRAWL: scrapy crawl myspider -o data_file.json
class MyspiderSpider(scrapy.Spider):
    name = "myspider"
    start_urls = (manual_links['link'].to_list())
    
    def parse(self, response):
        
        title_text = response.css("h1::text").get()
        
        for p_text in response.css("p::text").getall():
            
            yield {
                f"{title_text}": p_text
                }