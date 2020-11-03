# 2020_nlp

#### 아주아주 설명이 부실한 사용법
1. data 폴더에 다운받은 BalancedNewsCorpus 데이터를 넣는다
2. word_embeddings 폴더에 다운받은 embedding 파일들을 넣는다
4. requirments 들을 설치한다 (아래의 참고사항 확인)
3. train.py를 돌린다

#### 참고사항
* requirments.txt에 모든 디펜던시가 있지 않음, 아마 torch와 mecab은 알아서 잘 설치해야... mecab은 최신 버전을 직접 다운받아 설치하는걸 추천 (구버전 쓰니까 토크나이징이 다름;;)
* 아직 Word2Vec 300D token 밖에 구현이 안되어있음 (토크나이징까지 바꿔야해서... 생각보다 귀찮네..아오)
* 모델도 아직은 BiLSTM-max뿐!
* 현재 사용중인 preprocess는 공유받은 Word2Vec 노트북 파일에 코드를 그대로 가져옴
  * 해당 코드랑 약간의 사용법이 달라서 \<p>를 없애주는 부분만 추가
  * 이유는 알 수 없지만 현재 코드를 사용해서 나오는 token 중에서 약 30개? 정도는 pretrained embedding에 없음, 어떻게 열심히 세팅을 맞춰봐도 현재는 이게 최선
  * 해당 preprocess를 사용하면 문장구분이 싹다 사라진다는 엄청난 일이 발생함... 추후 수정예정
  * Glove는 preprocessing이 약간 또 다르다... 이넘들
