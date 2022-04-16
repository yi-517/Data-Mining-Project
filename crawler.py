# -*-coding:utf-8-*-
import requests
from lxml import html
import time
import pandas as pd

etree = html.etree


class BaiDu_Spider(object):
    def __init__(self, keyword):
        self.base_url = 'https://cn.bing.com/search?q={}'
        self.keyword = keyword
        self.url = self.base_url.format(self.keyword) + '&first={}'

    def get_html(self, page):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0'
        }
        try:
            r = requests.get(self.url.format(page), headers=headers)
            r.encoding = 'utf-8'
            res = etree.HTML(r.text)
            selector = res.xpath('//div[@id="b_content"]/main/ol/li[@class="b_algo"]')
            data_list = []
            for data in selector:
                item = {}
                item['title'] = ''.join(data.xpath('./div[@class="b_title"]/h2/a/text()'))
                item['abstract'] = ''.join(data.xpath('./div[@class="b_caption"]/p/text()'))
                item['link'] = ''.join(data.xpath('./div[@class="b_title"]/h2/a/@href'))
                data_list.append(item)
            return data_list, True
        except:
            pass

    def save(self, df, path):
        df.to_csv(path, header=None, index=False, encoding='utf-8')

def main():
    n = 1
    df = pd.DataFrame()
    while n < maxPage:
        data_list, flag = spider.get_html(n)

        for data in data_list:
            # spider.save_data(data)
            df = df.append(data, ignore_index=True)
            print(data)
        time.sleep(3)
        if flag is True:
            n = n + 10
        else:
            print('程序已经退出，n={}......'.format(n))
            break
    path = 'results_' + keyWord + '.csv'
    spider.save(df, path)


if __name__ == '__main__':
    maxPage = 500
    keyWord = ""
    spider = BaiDu_Spider(keyWord)
    main()


