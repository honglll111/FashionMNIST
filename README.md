# FashionMNIST
## Data 불러오기
```
# 라이브러리 import
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
```
```
# FashionMNIST 데이터 불러오기
fashion_mnist = keras.datasets.fashion_mnist  # fashion_mnist 데이터 로드
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # train set과 test set을 분리해서 로드
is_normalized = False  # 0부터 255 사이의 값을 255로 나누어 0과 1 사이의 값으로 정규화 했는지 확인 위해
```
## 데이터 구성 확인
```
# Train set 이미지 개수
print(train_images.shape[0])  # 60000
```
```
# Test set 이미지 개수
print(test_images.shape[0])  # 10000
```
```
# 이미지 크기
print(train_images.shape[1], train_images.shape[2])  # 28 28
```
```
# Train label 저장 형태
print(str(train_labels[:]))   # [9 0 0 ... 3 0 5]
```
```
# 이미지 값 예시
print(train_images[0])
```
![image](https://github.com/user-attachments/assets/ed7c24b9-3a6c-4d79-94fc-4bb05cdb4f93)
```
plt.figure()  # 그림 입력 준비
plt.imshow(train_images[0])  # Train set의 첫 번째 이미지
plt.colorbar  # 컬러바
plt.show()  # 이미지 출력
```
![image](https://github.com/user-attachments/assets/4d183d84-cd11-427f-b9c1-cf5a23037fed)

|Label|Description|
|------|---|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|
```
# Train set의 첫 10장의 이미지와 각각의 카테고리 확인
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10, 10))  # 그림 크기 10 X 10
for i in range(10):
  plt.subplot(1, 10, i+1)  
  plt.grid(False)  # 격자 off
  plt.imshow(train_images[i], cmap=plt.cm.binary)  # cmap(컬러맵)은 binary(회색조)
  plt.xlabel(class_names[train_labels[i]])
plt.show()  # 이미지 출력
```
![image](https://github.com/user-attachments/assets/c2a11bd1-fbab-4527-9cf1-86466d32faae)
## 정규화
```
if not is_normalized:  # is_normalized 변수가 False라면
  train_images = train_images / 255.0  # 0 ~ 255 -> 0 ~ 1
  test_images = test_images / 255.0  # 0 ~ 255 -> 0 ~ 1

  is_normalized = True 
```
## 모델 생성
```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # (28, 28) Flatten -> 784
     keras.layers.Dense(512, activation=tf.nn.tanh), # 784 -> 512, 활성함수는 tanh, 은닉층
    keras.layers.Dense(512, activation=tf.nn.tanh), # 512 -> 512, 활성함수는 tanh, 은닉층
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 클래스 개수인 10개로 변환, softmax 함수를 통해 확률값 반환
])
```
## 모델 훈련
```
model.compile(optimizer='adam',  # adam optimozer 사용
              loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy 손실함수 사용
              metrics=['accuracy'])  # accuracy로 평가
```
```
history = model.fit(train_images, 
                    train_labels,
                    epochs=10,
                    validation_data=(test_images, test_labels)  # Test set
                    )   
```
```
def plot_loss(history):
  plt.figure(figsize=(16, 10))  # 그림 크기 = (16, 10)
  plt.plot(history.epoch, history.history['val_loss'], '--', label='Test')  # validation loss를 에폭마다 점선으로 표시
  plt.plot(history.epoch, history.history['loss'], label='Train')  # training loss를 에폭마다 실선으로 표시

  plt.xlabel('Epochs')  # x축
  plt.ylabel('Loss')  # y축
  plt.legend()  # 범례 표시

  plt.xlim([0, max(history.epoch)])  # x축은 0부터 epoch 최댓값까지

plot_loss(history)  # history 출력
```
![image](https://github.com/user-attachments/assets/db532628-ba2b-44c1-a228-66c1a81767e2)
## 모델 성능 평가
```
def eval_model(model):
  test_loss, test_acc = model.evaluate(test_images, test_labels)  # Test set으로 모델 평가
  print('Test accuracy:', test_acc)  # 정확도 출력

eval_model(model)
```
![image](https://github.com/user-attachments/assets/3fc88b3a-2d3a-478d-9564-229826afe050)





