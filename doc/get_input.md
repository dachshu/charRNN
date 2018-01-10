# batch process를 위한 전처리
Input File을 모두 읽으면서 각 트윗이 시작하는 file pointer 위치를 리스트에 저장한다.
각 트윗의 끝은 '\n\n' 이다.

# class BatchReader
입력 데이터의 일부분만 가져오기 위한 클래스
## next_batch(self, size)
처음부터 순차적으로 size 만큼의 tweet data를 반환.  
남은 tweet의 수가 size보다 작은 경우 남은 모든 tweet을 반환.  
파일의 끝에 도달한 경우엔 빈 list 반환.
## random_batch(self, size)
size 만큼의 tweet data를 무작위로 선택해서 반환.  
전체 tweet의 수가 size보다 작은 경우, 모든 tweet을 반환.
## reset(self)
다음에 읽어야 할 tweet의 index를 초기화시켜 다시 처음부터 읽게 함. random_batch 함수를 이용하는 경우에는 의미 없음.