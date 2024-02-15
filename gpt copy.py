import pandas as pd
import datetime as dt
import tiktoken
from tqdm import tqdm
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from urllib import parse
import re


def sentiment(prompt, model='gpt-3.5-turbo'):
    """gpt 모델 설정"""
    
    client = OpenAI(api_key="발급 받은 api_key")
    response = client.chat.completions.create(
    model = model,
    messages=[
        {
        "role": "system",
        "content": "너는 재무전문가야. 주식 종목 추천에 특화되어 있어. 뉴스 기사를 보고 주가에 긍정적인 뉴스면 '긍정', 부정적인 뉴스면 '부정', 확실하지 않으면 '중립'으로 분류해줘."
        },
        {
        "role": "user",
        "content": prompt
        }
    ],
    temperature=0,
    max_tokens=64,
    top_p=1
    )
    return response.choices[0].message.content

def targetnews(corp_name, startdate, enddate):
    """네이버 금융 뉴스검색 사이트에서 스크레핑 했습니다. startdate, enddate 형식은 %Y-%m-%d (2024-01-15) 입니다."""

    corp_encoding = parse.quote_plus(corp_name, encoding='euc-kr')
    webpage = requests.get('https://finance.naver.com/news/news_search.naver?rcdate=&q='+corp_encoding+'&sm=title.basic&pd=3&stDateStart='+startdate+'&stDateEnd='+enddate) 
    result = BeautifulSoup(webpage.content, 'html.parser')

    newslist = result.select_one('dl.newsList')
    articlesubject = newslist.select('.articleSubject')
    articlesummary = newslist.select('.articleSummary')
    temp = []
    for ii, i in enumerate(articlesubject):
            temp.append(i.text.strip())
            # print(re.sub('\s{2,}','', articlesummary[ii].text.strip()))
            # print("")

    return temp

def sentiment_analysis_randomly(df, startdate, enddate, num = 100):
    """기본값 100개로 렌덤하게 기업 뽑아서 startdate~ enddate까지의 뉴스제목 감성분석"""
    
    data = []
    for coname in tqdm(df.sample(num)['Coname']):
        news_list = targetnews(coname, startdate, enddate)
        sentiments = [sentiment(news) for news in news_list]
        
        for news, sentiment_value in tqdm(zip(news_list, sentiments)):
            data.append({'Coname': coname, 'newshead': news, 'sentiment': sentiment_value})

    result = pd.DataFrame(data)
    return result


def sentiment_analysis_totalcorp(df, startdate, enddate):
    """전체 기업 기업 뽑아서 startdate~ enddate까지의 뉴스제목 감성분석"""

    data = []
    for coname in tqdm(df['Coname']):
        news_list = targetnews(coname, startdate, enddate)
        sentiments = [sentiment(news) for news in news_list]
        
        for news, sentiment_value in tqdm(zip(news_list, sentiments)):
            data.append({'Coname': coname, 'newshead': news, 'sentiment': sentiment_value})

    result = pd.DataFrame(data)
    return result

def num_token(prompt, encoding_type='gpt-3.5-turbo'):
     import tiktoken
     encoding = tiktoken.encoding_for_model(encoding_type)
     tokens = encoding.encode(prompt)
     num_token = len(tokens)

     return num_token

def inputcost(prompt, encoding_type='gpt-3.5-turbo'):
     """input cost: $0.0015/1K tokens"""
     token_cost = num_token(prompt, encoding_type) * 0.0015/1000
     return round(token_cost, 5)

def outputcost(prompt, encoding_type='gpt-3.5-turbo'):
    """output cost: $0.0020/1K tokens"""
    token_cost = num_token(prompt, encoding_type) * 0.002/1000
    return round(token_cost, 5)

def gpt_cost(newshead, sentiment):
    input_cost = sum([inputcost(news) for news in newshead])
    output_cost = sum([outputcost(output) for output in sentiment])
    return input_cost+output_cost

####################################################
##################기본 세팅##########################
####################################################

enddate = dt.datetime.today()
startdate = enddate - dt.timedelta(days=7)
enddate = dt.datetime.strftime(enddate,'%Y-%m-%d')
startdate = dt.datetime.strftime(startdate,'%Y-%m-%d')

####################################################
####################################################
####################################################

data = pd.ExcelFile("C:\PythonProject\자동투자, gpt, 네이버뉴스 크롤링\gpt감성분석\KR MktCap 011124.xlsx")
df = data.parse(0)

result = sentiment_analysis_randomly(df, startdate, enddate, num=100)
print(result)
result.to_excel(enddate +' sentiment analysis.xlsx')

print(f"gpt 비용 : ${gpt_cost(result['newshead'], result['sentiment'])}")

####################################################
####################################################
####################################################

# result = sentiment_analysis_totalcorp(df, startdate, enddate)
# print(result)
# result.to_excel(enddate +' sentiment analysis2.xlsx')

