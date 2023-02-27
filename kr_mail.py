import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import konlpy
import nltk
import urlextract
import url
import ssl
import socket

from html import unescape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt, Kkma
from hanspell import spell_checker
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
nltk.download('punkt')
url_extractor = urlextract.URLExtract()  # url 추출


def html_to_plain_text(html):  # HTML을 일반 텍스트로 변환하는 함수
    text = re.sub('<head.*?>.*?</head>', "", html,
                  flags=re.M | re.S | re.I)  # head 섹션 삭제
    text = re.sub('<a\s.*?>', 'HYPERLINK', text, flags=re.M |
                  re.S | re.I)      # 모든 a 태그를 HYPERLINK 문자로 변환
    text = re.sub('<.*?>', "", text, flags=re.M |
                  re.S)                      # 나머지 HTML 태그 제거
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M |
                  re.S)                # 여러 개행 문자를 하나로 합침
    return unescape(text)


def email_to_text(email):        # 포멧에 상관없이 이메일을 입력받아 일반 텍스트를 출력하는 함수
    try:
        html = None
        for part in email.walk():
            ctype = part.get_content_type()       # 해당 메일의 type 반환
            if not ctype in ("text/plain", "text/html"):
                continue
            try:
                content = part.get_content()
            except:
                content = str(part.get_payload())
            if ctype == 'text/plain':
                return content
            else:
                html = content
        if html:
            return html_to_plain_text(html)
    except AttributeError as e:
        return email


## HTTPS 연결 요청 함수 ##
def https_connect(url):
    ctx = ssl.create_default_context()
    s = ctx.wrap_socket(socket.socket(), server_hostname=url)
    s.connect((url, 443))
    return s

# HTTP or HTTPS 를 확인 후 붙이는 함수


def attach_http(urls):
    # 소켓 연결 제한 시간 설정
    socket.setdefaulttimeout(2)

    urldata = urls

    scheme_s = "https://"
    scheme = "http://"

    convert_url = []
    failed_url = []
    for url in urldata:
        try:
            https_connect(url)
            url = scheme_s + url
            convert_url.append(url)
        except Exception as e:
            url = scheme + url
            failed_url.append(url)
            pass
    url = convert_url + failed_url
    return url


# url을 나타내는 정규표현식
re_url = re.compile(r'\shttps?://www\.[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\swww\.[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\shttps?://[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\s[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\shttps?://www\.[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\swww\.[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\shttps?://[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\s[a-z0-9-]+\.[a-z0-9]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\shttps?://www\.[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\swww\.[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\shttps?://[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\s[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\shttps?://www\.[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\swww\.[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\shttps?://[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\s[a-z0-9-]+\.[a-z0-9]+\.[a-z]{2,4}[/]?|'
                    '\shttps?://www\.[a-z0-9-]+\..[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\swww\.[a-z0-9-]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\shttps?://[a-z0-9-]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\s[a-z0-9-]+\.[a-z]{2,4}\/[a-z0-9.!"#%&()\-+\,/:;<>=?@{}`|~\[\]]+|'
                    '\shttps?://www\.[a-z0-9-]+\.[a-z]{2,4}[/]?|'
                    '\swww\.[a-z0-9-]+\.[a-z]{2,4}[/]?|'
                    '\shttps?://[a-z0-9-]+\..[a-z]{2,4}[/]?|'
                    '\s[a-z0-9-]+\.[a-z]{2,4}[/]?', flags=re.I)


def cleaning(mail):
    X_transformed = []
    all_url = []
    clean_text = email_to_text(mail) or ""  # HTML 관련 태그 제거

    # url 추출
    re_kor = re.compile(r'[가-힣]+')  # 한글을 나타내는 정규표현식
    non_kor = re.sub(re_kor, ' ', clean_text)
    urls1 = re.findall(re_url, non_kor)
    urls2 = list(set(url_extractor.find_urls(non_kor)))
    urls = urls1 + urls2
    urls = ' '.join(s for s in urls)
    urls = list(set(url_extractor.find_urls((urls))))
    all_url.append(urls)
    for url in urls:
        clean_text = clean_text.replace(url, "")

    # 기본적인 전처리
    clean_text = re.sub(r'[^ ㄱ-ㅣ가-힣]', '', clean_text)
    clean_text = clean_text.lower()
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # 띄어쓰기, 맞춤법 검사
    spelled_sent = spell_checker.check(clean_text)
    clean_text = spelled_sent.checked

    # 형태소 분리
    okt = konlpy.tag.Okt()
    rmv_morpheme = []
    for word in okt.pos(clean_text, stem=True):
        if word[1] in ['Noun', 'Verb', 'Adjective']:
            rmv_morpheme.append(word[0])
    clean_text = ' '.join(rmv_morpheme)

    # 불용어 제거
    df = pd.read_csv('stopword.txt', header=None)
    df[0] = df[0].apply(lambda x: x.strip())
    stopwords = df[0].to_numpy()
    rmv_stopword = []
    for word in nltk.tokenize.word_tokenize(clean_text):
        if word not in stopwords:
            rmv_stopword.append(word)
    clean_text = ' '.join(rmv_stopword)

    X_transformed.append(clean_text)

    return np.array(X_transformed), all_url


def predict(email):
    MAX_LEN = 512
    X_NLP, URL = cleaning(email)
    url_in_mail = URL
    # 메일 내에 URL이 존재하는 경우
    if URL:
        url_pred = []
        http_url = []
        not_http_url = []
        for i in URL[0]:  # URL에 HTTP or HTTPS 가 존재하는지 확인한 후 붙이는 작업
            url_tmp = i.split('/')
            if ("https:" in url_tmp[0].lower()) or ("http:" in url_tmp[0].lower()):
                url_tmp[0] = url_tmp[0].lower()
                url_tmp[2] = url_tmp[2].lower()
                i = '/'.join(url_tmp)
                http_url.append(i)
            else:
                url_tmp[0] = url_tmp[0].lower()
                i = '/'.join(url_tmp)
                not_http_url.append(i)

        URL = http_url + attach_http(not_http_url)

        # 메일에 포함된 URL이 피싱인지 확인
        for i in URL:
            url_pred.append(url.url_check(i))

    X_NLP = ''.join(X_NLP)
    # 메일에 [CLS] 와 [SEP] 추가
    sentences = "[CLS] " + str(X_NLP) + " [SEP]"

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased", do_lower_case=False)
    tokenized_texts = tokenizer.tokenize(sentences)

    # 정수 인코딩
    input_ids = [tokenizer.convert_tokens_to_ids(tokenized_texts)]

    # 토큰화 시킨 배열이 최대 길이를 넘었을 경우 마지막에 SEP 토큰 추가
    if len(input_ids[0]) > 512:
        input_ids[0][511] = 102
    # 제로 패딩
    input_ids = pad_sequences(
        input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_labels = []
    for t in input_ids:
        seq_mask = [0 for i in seq]
        input_labels.append(seq_mask)

    # 예측 데이터 tensor로 변환
    test_inputs = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_masks)
    test_labels = torch.tensor(input_labels)

    # DataLoader로 변환
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data)

    input, masks, labels = next(iter(test_dataloader))

    # model에 입력할 값들, URL 예측값, 메일에 포함된 URL 반환
    return input, masks, labels, url_pred, url_in_mail
