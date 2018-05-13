from bs4 import BeautifulSoup
import xlwt
import progressbar
import requests
import random


def geturl(filename,url):
    user_agent = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
        'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
        'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50']
    headers = {"user-agent": "{}".format(random.choice(user_agent)), }
    url_list = []
    url = url

    page = requests.get(url,headers=headers)

    soup = BeautifulSoup(page.text,'lxml')
    item = soup.find_all('div',{'class':'item'})
    for i in item:
        try:
            url = i.find('a',{'class':'picRind'})['href'][2:]
        except:
            pass
        url_list.append(url)
    wb = xlwt.Workbook(encoding='utf-8')
    sh = wb.add_sheet('urllist')
    for i in progressbar.progressbar(range(len(url_list))):
        sh.write(i,0,url_list[i])
    wb.save(filename+'.xls')
    print('down')

geturl('makeup_mask','https://www.aliexpress.com/wholesale?catId=200002604&initiative_id=AS_20180513021256&SearchText=mask')

