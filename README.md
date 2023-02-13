<img src="https://grm-project-template-bucket.s3.ap-northeast-2.amazonaws.com/organization/kw/logo/%E1%84%80%E1%85%AA%E1%86%BC%E1%84%8B%E1%85%AE%E1%86%AB%E1%84%83%E1%85%A2.png" width="200" height="70">

# AI를 이용한 스피어 피싱 메일 탐지 기술
광운대학교 산학연계 __KW_Shield팀__,
웹 실행 프로그램입니다.

구글 BERT base multilingual cased를 이용해 개발하였습니다

(https://github.com/google-research/bert/blob/master/multilingual.md)
<br> </br>
	

# Requirements
파이썬 프로그램 실행을 위해 필요한 패키지 참고하시기 바랍니다
* [requirements.txt](https://github.com/KW-shield/WebApplication/files/10611734/requirements.txt)
<br> </br>
# 학습 데이터 셋
* 한국어

라벨링한 메일 데이터셋 사용
| 구분 | __직접 추출__ |
| :---: | :---: |
| ham mail | 1200 |
| spam mail | 800 |

* 영어

| 구분 | __spamassassin__ |   __kaggle__   |
| :---: | :---: | :---: |
| ham mail | 6951 | 4516 |
| spam mail | 2398 | 653 |

<br> </br>
# 학습 환경
* 주 개발 언어    : __Python(Anaconda)__
* 딥러닝 라이브러리: __Pytorch__
* 학습 진행환경    : __Google Colab Pro/Pro+__
* 웹 프레임워크    : __Flask__

