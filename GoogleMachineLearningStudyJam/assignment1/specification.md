# 기계학습 공부 과제 1
## 간단한 Neural Network 구현(MNIST Dataset)

#### 2019.02.27. https://slack-files.com/TG8A44G6M-FGJ0CHMT4-6edc167dbd

### 이 과제를 통해 기계학습이나 데이터를 다루는 다른 작업에서 필요한 정도의 기초적 Python 실력을 키울 수 있습니다. 또한, 신경망이라는 것의 아주 원시적인 뼈대가 어떻게 되어있는지 바닥부터 이해해볼 기회입니다.

1. '밑바닥부터 시작하는 딥러닝' 100쪽에 있는 ch03/neuralnet_mnist.py를 참고하여 다음의 간단한 3층 신경망(2층이라고 하기도 함, 64쪽 참고)을 완성하세요. (책의 코드는 https://github.com/WegraLee/deep-learning-from-scratch 에서 다운로드 가능! 이 과제와 별개로 책을 보면서 적극 활용하세요!)
2. 샘플 데이터는 5000*13 배열로 임의의 값이 채워져 있는 파일입니다. 이 파일의 마지막 열(13열)은 12열에 대한 0-9 사이의 정답 숫자로 되어 있습니다.(즉, 이 학습은 Supervised Learning(감독 학습, 지도 학습)) 이 파일을 읽어서 x_train, y_train으로 불러들입니다. 즉, x_train은 (5000, 12) 형태의 데이터 배열이고 y_train은 (5000, 1) 형태의 정답 배열입니다.
3. 이 simple forward network는 손실 함수도 구현되지 않은 기본 단계라 당연히 성능이 낮을 것입니다. 앞으로 조금씩 성능을 향상시켜보도록 합시다. 우선 skeleton code(뼈대 코드)에 제시된 함수들을 완성하여 이 모델이 작동되어 정확도를 출력하도록 하세요.
4. keras, sklearn, pandas 등의 패키지에서 제공하는 모듈을 쓰지말고, numpy 등의 기본적인 툴만 사용하세요!
5. Python 등 프로그래밍 언어에 생소하신 분들은 어려울 것이 당연합니다. 하지만 직접 맞닥뜨려서 멘땅에 헤딩하는 것이 가장 효율적인 습득법이므로, 질문과 검색을 많이 하면서 어느 정도 시간을 할애해보시는 것을 추천드립니다! 사실 직접 작성해야 하는 것이 얼마 없고 꽤 간단한 편이지만, 기존에 작성되어 있는 코드를 이해하는 것도 중요하고 어려운 일입니다. 너무 기초적인 것도 감을 못 잡겠다, 그래도 괜찮으니 질문해주세요!