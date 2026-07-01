from konlpy.tag import Okt
import os

okt = Okt()

wordlist = okt.morphs("파이썬을 활용한 한글 형태소 분석 입니다.",norm=True, stem=True)
print(wordlist)
