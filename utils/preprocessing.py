import re
import hanja
from konlpy.tag import Mecab

tokenizer = Mecab()
removal_list =  "‘, ’, ◇, ‘, ”,  ’, ', ·, \“, ·, △, ●,  , ■, (, ), \", >>, `, /, -,∼,=,ㆍ<,>, .,?, !,【,】, …, ◆,%"

EMAIL_PATTERN = re.compile(r'''(([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)(\.[a-zA-Z]{2,4}))''', re.VERBOSE)
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.VERBOSE)
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)

def cleansing_other(sentence: str = None) -> str:
    """
    문장을 전처리 (이메일, URL, 공백 등 제거) 하는 함수
    :param sentence: 전처리 대상 문장
    :return: 전처리 완료된 문장
    """
    sentence = re.sub(EMAIL_PATTERN, ' ', sentence)
    sentence = re.sub(URL_PATTERN, ' ', sentence)
    sentence = re.sub(MULTIPLE_SPACES, ' ', sentence)
    sentence = sentence.replace(", )", "")
    
    return sentence


def cleansing_chinese(sentence: str = None) -> str:
    """
    한자를 변환하는 전처리를 하는 함수
    :param sentence: 전처리 대상 문장
    :return: 전처리 완료된 문장
    """
    # chinese character를 앞뒤로 괄호가 감싸고 있을 경우, 대부분 한글 번역임
    sentence = re.sub("\([\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD]+\)", "", sentence)
    # 다른 한자가 있다면 한글로 치환
    if re.search("[\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD]", sentence) is not None:
        sentence = hanja.translate(sentence, 'substitution')

    return sentence

def cleansing_special(sentence: str = None) -> str:
    """
    특수문자를 전처리를 하는 함수
    :param sentence: 전처리 대상 문장
    :return: 전처리 완료된 문장
    """
    sentence = re.sub("[.,\'\"’‘”“!?]", "", sentence)
    sentence = re.sub("[^가-힣0-9a-zA-Z\\s]", " ", sentence)
    sentence = re.sub("\s+", " ", sentence)
    
    sentence = sentence.translate(str.maketrans(removal_list, ' '*len(removal_list)))
    sentence = sentence.strip()
    
    return sentence

def cleansing_numbers(sentence: str = None) -> str:
    """
    숫자를 전처리(delexicalization) 하는 함수
    :param sentence: 전처리 대상 문장
    :return: 전처리 완료된 문장
    """
    
    sentence = re.sub('[0-9]+', 'NUM', sentence)
    sentence = re.sub('NUM\s+', "NUM", sentence)
    sentence = re.sub('[NUM]+', "NUM", sentence)
    
    return sentence

def preprocess_sent(sentence: str = None) -> str:
    """
    모든 전처리를 수행 하는 함수
    :param sentence: 전처리 대상 문장
    :return: 전처리 완료된 문장
    """
    sentence = sentence.replace('<p>',' ').replace('</p>','<sep>')
    sent_clean = sentence
    sent_clean = cleansing_other(sent_clean)
    sent_clean = cleansing_chinese(sent_clean)
    sent_clean = cleansing_special(sent_clean)
    sent_clean = cleansing_numbers(sent_clean)
    sent_clean = re.sub('\s+', ' ', sent_clean)

    return sent_clean


# Glove preprocessing
def cleaning_strings(input_text):

    input_text = input_text.replace('<p>',' ').replace('</p>','\n')  # 문단 간 구분이 필요 없으므로, 문단 구분자 삭제, 줄바꿈 삽입 
    input_text = input_text.translate(str.maketrans('①②③④⑤⑥⑦⑴⑵⑶⑷⑸ⅠⅡⅢ','123456712345123'))  # 숫자 정리
    input_text = input_text.translate(str.maketrans('―“”‘’〉∼\u3000', '-""\'\'>~ '))  # 유니코드 기호 정리
    input_text = input_text.translate({ord(i): None for i in '↑→↓⇒∇■□▲△▶▷▼◆◇○◎●★☆☞♥♪【】'})  # 특수문자 정리

    # 이메일 패턴 제거
    EMAIL_PATTERN = re.compile(r'(([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)(\.[a-zA-Z]{2,4}))')
    input_text = re.sub(EMAIL_PATTERN, ' ', input_text)

    # url 패턴 제거
    URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    input_text = re.sub(URL_PATTERN, ' ', input_text)
    
    input_text = re.sub('\(.+?\)','',input_text)  # 괄호 안 내용 삭제
    input_text = hanja.translate(input_text, 'substitution')  # 한자 -> 한글 치환
    input_text = re.sub('([\'\",.\(\)\[\]\{\}<\>\:\;\/\?\!\~\…\·\=\+\-\_])',' \g<1> ',input_text)  # 각종 문장부호 전후 띄어쓰기


    while True:
        temp_text = re.sub('(.+)([.|,])([가-힣]+)','\g<1>\g<2> \g<3>', input_text)  # 앞텍스트.뒷텍스트  처럼 마침표/쉼표 뒤에 띄어쓰기가 없는 경우 띄어쓰기
        if input_text == temp_text:                         # -> 재귀적으로 구현하여 여러번 시행
            input_text = temp_text
            break
        else:
            input_text = temp_text[:]

    input_text = re.sub('[0-9]+','NUM',input_text)  # 모든 숫자 NUM 으로 마스킹      
    input_text = re.sub('[ ]{2,}',' ', input_text)  # 띄어쓰기 2번 이상 중복된 경우 하나로 통합

    output_text = input_text.strip()

    return output_text

def glove_tokenizer(text):
    temp_tks = tokenizer.pos(text)
    result = [f'{x[0].strip()}/{x[1].strip()}' for x in temp_tks if x[0]!=''] 
    return result