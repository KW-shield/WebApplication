import torch
import numpy as np
import tensorflow as tf
import nltk
import urlextract
import re
import url
import ssl
import socket

from html import unescape
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


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


stemmer = nltk.PorterStemmer()  # 어간 추출
url_extractor = urlextract.URLExtract()  # url 추출
nltk.download('stopwords')  # 불용어
stop_words = stopwords.words('english')

re_space = re.compile(r'[\W+_]')  # 스페이스바를 나타내는 정규표현식
re_delete_numberspot = re.compile(r'[,]')  # ','을 나타내는 정규표현식

# re_arrange_email과 re_email을 사용하여 이메일을 나타내는 정규표현식
re_arrange_email = re.compile(r'[^\w.@-]')
re_email = re.compile('[a-zA-Z0-9+-\_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')


def preprocess(mail):
    X_transformed = []
    all_url = []  # 메일안의 url을 저장하는 list

    text = email_to_text(mail) or ""  # HTML 관련 태그 제거

    text = re.sub(re_delete_numberspot, '', text)  # 숫자 사이의 ',' 제거

    # url 추출
    urls = list(set(url_extractor.find_urls(text)))
    urls.sort(key=lambda url: len(url), reverse=True)
    all_url.append(urls)
    # 해당 url을 모두 ""로 변경
    for url in urls:
        text = text.replace(url, "")

    text = text.lower()  # 모든 문자를 소문자화

    # 이메일 추출
    text = re.sub(re_arrange_email, ' ', text)
    # 해당 email을 모두 ""로 변경
    text = re.sub(re_email, '', text)

    # 숫자와 문자를 제외한 것을 띄어쓰기로 변경
    text = re.sub(re_space, ' ', text)
    text = ' '.join(text.split())

    # 불용어 제거
    text = text.split()
    text = [word for word in text if not word in stop_words]

    # 길이가 2보다 작은 단어 제거
    text = [word for word in text if len(word) > 2]

    # 어간 추출
    text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)

    X_transformed.append(text)

    return np.array(X_transformed), all_url

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


def predict(email):
    MAX_LEN = 512
    X_NLP, URL = preprocess(email)
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

    # 메일에 [CLS] 와 [SEP] 추가
    X_NLP = ''.join(X_NLP)
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
