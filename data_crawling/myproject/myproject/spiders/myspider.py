import scrapy


class MyspiderSpider(scrapy.Spider):
    name = "myspider"
    start_urls = ([
        "example.com",
        'https://vietnam.vnanet.vn/english/tin-van/vietnam-faces-challenges-of-aging-society-316541.html',
        'https://vietnam.vnanet.vn/english/long-form/breakthrough-in-science-technology-innovation-and-digital-transformation-389737.html'
        ])
    

    def parse(self, response):
        for title in response.css("h1::text").getall():
            yield {"title": title}
