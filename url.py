import re
import whois
import socket
import requests
import urllib.parse
import pandas as pd
import numpy as np
import pickle
import whois_parser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tld import get_tld

# 1 : 정상 / -1 : 피싱
# ip주소를 쓰는지 확인인


def having_IP_Address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
        # IPv4 in hexadecimal
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
        '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)  # Ipv6
    if match:
        return -1
    else:
        return 1

# 1 : 정상 / 0 : 의심 / -1 : 피싱
# url의 길이


def URL_Length(url):
    if len(url) < 54:
        return 1
    elif len(url) >= 54 and len(url) <= 75:
        return 0
    else:
        return -1

# 1 : 정상 / -1 : 피싱
# 단축 url 확인


def Shortining_Service(url):
    match = re.search('[.|/]shorturl[.]at|[.|/]shortto[.]com|[.|/]shorte[.]st|[.|/]go2l[.]ink|[.|/]x[.]co|[.|/]t[.]co|'
                      '[.|/]tinyurl[.]com|[.|/]is[.]gd|[.|/]cli[.]gs|[.|/]yfrog[.]com|[.|/]migre[.]me|[.|/]ff[.]im|'
                      '[.|/]tiny[.]cc|[.|/]url4[.]eu|[.|/]twit[.]ac|[.|/]su[.]pr|[.|/]twurl[.]nl|[.|/]snipurl[.]com|'
                      '[.|/]short[.]to|[.|/]BudURL[.]com|[.|/]ping[.]fm|[.|/]post[.]ly|[.|/]Just[.]as|[.|/]bkite[.]com|'
                      '[.|/]snipr[.]com|[.|/]fic[.]kr|[.|/]loopt[.]us|[.|/]doiop[.]com|[.|/]short[.]ie|[.|/]kl[.]am|'
                      '[.|/]wp[.]me|[.|/]rubyurl[.]com|[.|/]om[.]ly|[.|/]to[.]ly|[.|/]bit[.]do|[.|/]lnkd[.]in|'
                      '[.|/]han[.]gl|[.|/]db[.]tt|[.|/]qr[.]ae|[.|/]adf[.]ly|[.|/]goo[.]gl|[.|/]bitly[.]com|[.|/]cur[.]lv|'
                      '[.|/]ow[.]ly|[.|/]bit[.]ly|[.|/]ity[.]im|[.|/]q[.]gs|[.|/]po[.]st|[.|/]bc[.]vc|[.|/]twitthis[.]com|'
                      '[.|/]u[.]to|[.|/]j[.]mp|[.|/]buzurl[.]com|[.|/]cutt[.]us|[.|/]u[.]bb|[.|/]yourls[.]org|[.|/]x[.]co|'
                      '[.|/]prettylinkpro[.]com|[.|/]scrnch[.]me|[.|/]filoops[.]info|[.|/]vzturl[.]com|[.|/]qr[.]net|'
                      '[.|/]1url[.]com|[.|/]tweez[.]me|[.|/]v[.]gd|[.|/]tr[.]im|[.|/]link[.]zip[.]net|[.|/]shorl[.]com|'
                      '[.|/]ad[.]vu|[.|/]smallurl[.]co|[.|/]shor7[.]com|[.|/]r2me[.]com|[.|/]bl[.]lnk|[.|/]hurl[.]ws|'
                      '[.|/]poprl[.]com|[.|/]me2[.]do|[.|/]wo[.]gl|[.|/]wa[.]gl',
                      url)
    if match:
        return -1
    else:
        return 1

# 1 : 정상 / -1 : 피싱
# '@'가 있는지


def having_At_Symbol(url):
    if '@' in url:
        return -1
    else:
        return 1

# 1 : 정상 / -1 : 피싱
# '-'가 있는지


def prefixSuffix(url):
    if '-' in urllib.parse.urlparse(url).netloc:
        return -1
    else:
        return 1

#  1 : 정상 / -1 : 피싱
# '//'가 있는지


def double_slash_redirecting(url):
    parse = urllib.parse.urlparse(url)
    path = parse.path
    if '//' in path:
        return -1
    else:
        return 1


match_www = re.compile('www[0-9]*[.]')

# www.을 지우는 함수


def remove_www(url):
    if "https://" in url[:8]:
        if match_www.match(url[8:14]):
            url = re.sub(match_www, "", url, count=1)
    elif "http://" in url[:7]:
        if match_www.match(url[7:13]):
            url = re.sub(match_www, "", url, count=1)
    else:
        if match_www.match(url[:6]):
            url = re.sub(match_www, "", url, count=1)
    return url

# 1 : 정상 / 0 : 의심 / -1 : 피싱
# 서브 도메인의 개수 확인


def having_Sub_Domain(url, ip_address):
    url = remove_www(url)
    if ip_address == 1:
        domain = get_tld(url, as_object=True, fix_protocol=True)
        if domain.subdomain == "":
            return 1
        dot = domain.subdomain.count('.')
        if dot == 0:
            return 0
        else:
            return -1
    else:
        return -1

# 1 : 정상 / -1 : 피싱
# 도메인 사용 기간을 나타내는 함수


def Domain_registration_length(url):
    try:
        domain = whois.whois(url)
        if type(domain.expiration_date) is list:
            expiration_date = domain.expiration_date[0]
        else:
            expiration_date = domain.expiration_date

        if type(domain.updated_date) is list:
            updated_date = domain.updated_date[0]
        else:
            updated_date = domain.updated_date

        total_date = (expiration_date - updated_date).days
        if total_date <= 365:
            return -1
        else:
            return 1
    except:
        return -1


# 1 정상 / -1 피싱
# https의 유무 확인
def httpSecure(url):
    # It supports the following URL schemes: file , ftp , gopher , hdl ,
    htp = urlparse(url).scheme
    # http , https ... from urllib.parse
    match = str(htp)
    if match == 'https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return -1


def add_www(url):
    if "https://" in url[:8]:
        if not match_www.match(url[8:14]):
            url = url[:8] + "www." + url[8:]
    elif "http://" in url[:7]:
        if not match_www.match(url[7:13]):
            url = url[:7] + "www." + url[7:]
    else:
        if not match_www.match(url[:6]):
            url = "www." + url
    return url

# HTTPS 검사, domain으로 받기
# 1 정상 / -1 피싱
# 포트가 열려있는지 확인


def port_scan(url):
    if "http://" in url[:7]:
        return -1

    try:
        res = get_tld(url, as_object=True, fix_protocol=True)
        domain = res.parsed_url.netloc
        ip = socket.gethostbyname(domain)
    except:
        return -1

    socket.setdefaulttimeout(0.3)

    # 80 : http, 443 : https
    ports = [443, 21, 22, 23, 445, 1433, 1521, 3306, 3389]
    for port in ports:
        s = socket.socket()
        if port == 443:
            try:
                s.connect((ip, port))
                s.close()
            except:
                return -1
        else:
            try:
                s.connect((ip, port))
                s.close()
                return -1
            except:
                pass
    return 1


def featureExtraction(data2):
    data2['having_IP_Address'] = data2['url'].apply(
        lambda i: having_IP_Address(i))
    data2['URL_Length'] = data2['url'].apply(lambda i: URL_Length(i))
    data2['Shortening_Service'] = data2['url'].apply(
        lambda i: Shortining_Service(i))
    data2['having_At_Symbol'] = data2['url'].apply(
        lambda i: having_At_Symbol(i))
    data2['prefixSuffix'] = data2['url'].apply(lambda i: prefixSuffix(i))
    data2['double_slash_redirecting'] = data2['url'].apply(
        lambda i: double_slash_redirecting(i))
    data2['having_Sub_Domain'] = data2.apply(
        lambda x: having_Sub_Domain(x['url'], x['having_IP_Address']), axis=1)
    data2['httpSecure'] = data2['url'].apply(lambda i: httpSecure(i))
    data2['Domain_registration_length'] = data2['url'].apply(
        lambda i: Domain_registration_length(i))
    data2['port_scan'] = data2['url'].apply(lambda i: port_scan(i))
    return data2

# 1 정상 / -1 피싱
# html 코드 안에 해당 도메인이 존재하는지 확인하는 함수


def Domain_in_Source(response, url):
    try:
        url = remove_www(url)
        res = get_tld(url, as_object=True, fix_protocol=True)
        domain = res.parsed_url.netloc
    except:
        return -1
    if response == "":
        return -1
    else:
        if str(domain) in response.text:
            return 1
        else:
            return -1

# 1 정상 / 0 의심 / -1 피싱
# html 코드의 길이를 확인하는 함수


def Length_of_Source(response):
    if response == "":
        return -1
    else:
        if len(response.text) < 5000:
            return -1
        elif 5000 <= len(response.text) < 50000:
            return 0
        else:
            return 1

# 1 정상 / -1 피싱
# html 코드에 head 태그나 body 태그가 2개 이상 존재하는지 확인하는 함수


def duplicated_HEAD(response):
    if response == "":
        return -1
    else:
        if len(re.findall(r"<head>", response.text)) >= 2:
            return -1
        elif len(re.findall(r"<body>", response.text)) >= 2:
            return -1
        else:
            return 1

# script 태그에서 src의 비율을 확인하는 함수


def External_Load_Script(response):
    if response == "":
        return -1
    else:
        src_tag_num = 0
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all("script")
        if len(links) == 0:
            return 0
        for link in links:
            try:
                if link['src']:
                    src_tag_num += 1
            except:
                pass
        if len(links) == src_tag_num:
            return -1
        else:
            return 1


def htmlExtraction(url):
    features = []
    # HTML & Javascript based features (4)
    try:
        response = requests.get(url, timeout=5)
    except:
        response = ""
    features.append(Domain_in_Source(response, url))
    features.append(Length_of_Source(response))
    features.append(duplicated_HEAD(response))
    features.append(External_Load_Script(response))

    return features


def url_check(url):

    data = pd.DataFrame()
    data['url'] = pd.Series(url)

    data = featureExtraction(data)

    feature = []
    for url in data['url']:
        feature.append(htmlExtraction(url))

    feature_name = ['Domain_in_Source', 'Length_of_Source',
                    'duplicated_HEAD', 'External_Load_Script']
    data[[i for i in feature_name]] = pd.DataFrame(
        [i for i in feature], index=data.index)

    test_data = data.drop(['url'], axis=1)

# 모델 불러오기
    with open('url_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)

    pred = loaded_model.predict_proba(test_data)
    return pred
