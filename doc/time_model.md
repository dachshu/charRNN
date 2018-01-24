# Time Model
트윗의 시간 기록을 이용해서 시간 간격을 학습하고, 다음 시간을 예측하는 모델. 

트윗들이 작성된 unix time들 사이의 간격과 하루 안에서 트윗들이 작성된 시간을 학습한다.
## 요구사항
- [python](https://www.python.org/) 3.5 이상
- [tensorflow](https://www.tensorflow.org/) for python
- [numpy](www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
## 사용법
시간 기록을 이용해 모델을 학습시키려면 다음 명령어를 입력한다.
```
python3 time_train.py LOG_FILE
```
`time_train.py` 모듈을 실행시킬 때 `--seq_length`, `--epoch`, `--learning_rate` 등의 옵션을 추가로 줄 수 있다. 모든 옵션에 대한 설명은 `-h` 혹은 `--help` 옵션으로 볼 수 있다.

`LOG_FILE`은 트윗 작성 시간이 unix time으로 한 줄에 하나씩 기록되어 있는 파일을 말한다. 이 파일에는 최소 11개의 트윗에 대한 작성 시간이 기록되어 있어야 한다.
학습된 모델은 `time_save` 디렉터리에 저장된다.

학습된 모델을 통해 새로운 시간을 예측해 내기 위해선 다음 명령어를 사용한다.
```
python3 time_generate.py LOG_FILE
```

`time_generate.py`는 다른 python 모듈에서 import 해서 사용할 수 있도록 함수를 제공한다. 함수 이름은 `get_next_remaining_seconds`로 매개변수를 통해 `LOG_FILE`로 쓰일 파일의 이름을 전달할 수 있다. 기본값은 `time_log` 이다. 이 함수는 현재 시간과 예측된 시간을 비교해서 사용자가 다음 트윗을 올리기까지 얼마나 기다려야 하는지를 초 단위로 반환한다. 현재 시간이 예측된 시간보다 더 뒤라면 0을 반환한다.
## TODO
- [ ] LOG_FILE가 포함해야 하는 시간의 양이 학습된 모델의 sequence length에 맞춰 변화하도록 수정
- [ ] 중복되는 코드 제거
