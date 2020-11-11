# 2020_nlp

#### 아주아주 설명이 부실한 사용법
1. data 폴더에 다운받은 BalancedNewsCorpus 데이터를 넣는다
2. word_embeddings 폴더에 다운받은 embedding 파일들을 넣는다
3. requirments 들을 설치한다 (아래의 참고사항 확인)
4. train.py를 돌린다
* Best perform 모델은 해당 옵션을 사용함 --tokienizer mecab --dropout 0.7 --fintune-embedding 50 --rnn-type gru


#### 참고사항
* requirments.txt에 모든 디펜던시가 있지 않음, 아마 torch와 mecab은 알아서 잘 설치해야... mecab은 최신 버전을 직접 다운받아 설치하는걸 추천 (구버전 쓰니까 토크나이징이 다름;;)
* 계속 업데이트될 예정, 추가 구현은 언제나 환영 다만 별도의 브랜치를 사용해주세요 ㅠㅠ
* 궁금한게 있으면 언제나 카톡으로 질문 환영합니다
* ~아직 Word2Vec 300D token 밖에 구현이 안되어있음 (토크나이징까지 바꿔야해서... 생각보다 귀찮네..아오)~
* morph와 whitespace 버전 Word2Vec, FastText 사용가능 (주의할 부분은 사용하는 morph버전을 사용할 때는 tokenizer을 mecab으로 줘야함) 
* 모델도 아직은 BiLSTM-max뿐!
* attention이 모델에 추가됨 dot, bahdanau (bahdanau 버전이 현재 버전, dot을 쓰려면 코드를 수정해야함)
* 현재 사용중인 preprocess는 공유받은 Word2Vec 노트북 파일에 코드를 그대로 가져옴
  * 해당 코드랑 약간의 사용법이 달라서 \<p>를 sep로 변경하는 부분만 추가
  * 이유는 알 수 없지만 현재 코드를 사용해서 나오는 token 중에서 약 30개? 정도는 pretrained embedding에 없음, 어떻게 열심히 세팅을 맞춰봐도 현재는 이게 최선 (mecab 기준)
  * Glove는 preprocessing이 약간 또 다르다... 이넘들
* valid 따위 아직 split해서 만들지 않았고, 약간 cheating이지만 현재는 그냥 test를 사용해서 평가함
* ~아직 베이스라인에 가까운 모델이라 max 성능 약 80.5정도 나옴~
* 현재 최고의 성능을 보인건 bahdaunau lstm 2 layer버전 85.31
