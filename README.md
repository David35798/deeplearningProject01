# 스마트폰 중독 위험도 예측 시스템

스마트폰 사용 패턴 데이터를 기반으로 딥러닝 이진 분류 모델을 학습하고, 사용자의 스마트폰 중독 위험도를 Streamlit 웹 애플리케이션에서 예측할 수 있도록 구현한 프로젝트입니다.

스마트폰 사용 패턴 데이터를 기반으로 사용자의 스마트폰 중독 위험도를 예측하는 딥러닝 기반 이진 분류 프로젝트입니다.  
일일 사용시간, SNS 사용시간, 게임 사용시간, 수면 시간, 알림 수, 앱 실행 횟수 등의 변수를 활용하여 중독 여부를 예측하는 딥러닝 모델을 학습하고, Streamlit 웹 애플리케이션으로 구현했습니다.

---

## 프로젝트 소개

본 프로젝트는 스마트폰 사용 습관 데이터를 분석하여 사용자의 스마트폰 중독 위험도를 예측하는 딥러닝 프로젝트입니다.

사용자의 일일 스마트폰 사용시간, SNS 사용시간, 게임 사용시간, 수면 시간, 하루 알림 수, 앱 실행 횟수 등의 행동 패턴을 입력 변수로 사용하였고, `addicted_label` 값을 타깃으로 설정하여 스마트폰 중독 여부를 예측하는 모델을 학습했습니다.

학습된 딥러닝 모델은 Streamlit 웹 애플리케이션과 연동하여 사용자가 직접 스마트폰 사용 패턴을 입력하고, 예측된 중독도와 위험 단계를 확인할 수 있도록 구현했습니다.

단순히 모델을 학습하는 것에 그치지 않고, 데이터 전처리, 딥러닝 모델 설계, 모델 학습, 모델 저장, 예측 결과 해석, 웹 서비스 구현까지 딥러닝 프로젝트의 전체 흐름을 경험하는 것을 목표로 진행했습니다.

---

## 주요 기능

- 스마트폰 사용 패턴 기반 중독 위험도 예측
- 사용자 입력값 기반 실시간 예측 기능 제공
- 딥러닝 모델을 활용한 이진 분류 수행
- 예측 확률을 기반으로 4단계 위험도 분류
  - 정상
  - 관심군
  - 주의군
  - 고위험군
- 입력값 요약 정보 제공
- 중독도 확률 및 진행률 표시
- 예측 결과에 대한 간단한 해석 문구 제공
- 실제값과 예측값 비교 기능 제공
- 정확 / 오분류 결과 필터링
- 예측 결과 CSV 다운로드 기능 제공

---

## 사용 기술

### Language

- Python

### Data Processing / Machine Learning

- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Joblib

### Deep Learning

- Dense Layer
- Dropout
- Sigmoid
- Binary Classification

### Visualization / Analysis

- Matplotlib
- Seaborn
- Confusion Matrix
- ROC Curve

### Web Application

- Streamlit

---

## 데이터셋

본 프로젝트에서는 스마트폰 사용 및 중독 관련 데이터를 사용했습니다.

사용한 데이터 파일은 다음과 같습니다.

```text
Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
```

데이터는 총 7,500개의 행으로 구성되어 있으며, 스마트폰 사용 습관과 관련된 다양한 컬럼을 포함합니다.

주요 컬럼은 다음과 같습니다.

| 컬럼명 | 설명 |
|---|---|
| daily_screen_time_hours | 일일 스마트폰 사용시간 |
| social_media_hours | SNS 사용시간 |
| gaming_hours | 게임 사용시간 |
| sleep_hours | 수면 시간 |
| notifications_per_day | 하루 알림 수 |
| app_opens_per_day | 하루 앱 실행 횟수 |
| addicted_label | 스마트폰 중독 여부 |

---

## 데이터 전처리

모델 학습에 사용한 입력 변수는 다음과 같습니다.

```python
features = [
    'daily_screen_time_hours',
    'social_media_hours',
    'gaming_hours',
    'sleep_hours',
    'notifications_per_day',
    'app_opens_per_day'
]
```

타깃 변수는 다음과 같습니다.

```python
target = 'addicted_label'
```

입력 데이터는 학습 데이터와 테스트 데이터로 분리했습니다.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

딥러닝 모델의 안정적인 학습을 위해 `StandardScaler`를 사용하여 입력 변수를 표준화했습니다.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

학습이 끝난 뒤에는 웹 애플리케이션에서도 동일한 전처리를 적용할 수 있도록 scaler를 저장했습니다.

```python
joblib.dump(scaler, "model/scaler.pkl")
```

---

## 모델 구조

본 프로젝트에서는 Keras의 Sequential API를 사용하여 딥러닝 기반 이진 분류 모델을 구현했습니다.

모델 구조는 다음과 같습니다.

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

모델은 두 개의 Dense 은닉층과 Dropout 계층으로 구성했습니다.

마지막 출력층에서는 `sigmoid` 활성화 함수를 사용하여 0과 1 사이의 확률값을 출력하도록 구성했습니다.  
출력된 확률값을 기준으로 스마트폰 중독 가능성을 판단합니다.

---

## 모델 학습

모델은 이진 분류 문제에 적합한 `binary_crossentropy` 손실 함수를 사용하여 학습했습니다.

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

학습 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 50 |
| Batch Size | 16 |
| Validation Split | 0.2 |

학습이 완료된 모델은 웹 애플리케이션에서 재사용할 수 있도록 저장했습니다.

```python
model.save("model/addiction_model.h5")
```

---

## 모델 성능

학습 로그 기준 마지막 epoch의 성능은 다음과 같습니다.

| 지표 | 값 |
|---|---:|
| Train Accuracy | 0.9404 |
| Train Loss | 0.0999 |
| Validation Accuracy | 0.9317 |
| Validation Loss | 0.1169 |

또한 모델 평가를 위해 다음과 같은 시각화 자료를 생성했습니다.

- Accuracy 그래프
- Loss 그래프
- Confusion Matrix
- ROC Curve
- 실제값 vs 예측값 비교 그래프

이를 통해 모델의 학습 과정, 분류 성능, 실제값과 예측값의 차이를 확인할 수 있도록 구성했습니다.

---

## 위험도 분류 기준

모델은 0과 1 사이의 확률값을 출력합니다.  
해당 확률값을 기준으로 사용자의 스마트폰 중독 위험도를 4단계로 분류했습니다.

| 확률 구간 | 위험도 |
|---|---|
| 0.00 ~ 0.24 | 정상 |
| 0.25 ~ 0.49 | 관심군 |
| 0.50 ~ 0.74 | 주의군 |
| 0.75 ~ 1.00 | 고위험군 |

분류 함수는 다음과 같이 구성했습니다.

```python
def classify_risk(prob):
    if prob < 0.25:
        return "정상"
    elif prob < 0.50:
        return "관심군"
    elif prob < 0.75:
        return "주의군"
    else:
        return "고위험군"
```

---

## 웹 애플리케이션 구성

Streamlit을 사용하여 학습된 딥러닝 모델을 웹 애플리케이션 형태로 구현했습니다.

웹 애플리케이션은 크게 두 개의 탭으로 구성되어 있습니다.

### 사용자 예측 탭

사용자가 직접 스마트폰 사용 패턴을 입력하면 모델이 중독도를 예측합니다.

입력 항목은 다음과 같습니다.

- 일일 사용시간
- SNS 사용시간
- 게임 사용시간
- 수면 시간
- 알림 수
- 앱 실행 횟수

예측 결과로는 다음 정보를 제공합니다.

- 입력값 요약
- 중독도 확률
- 위험도 단계
- 위험도 기준
- 결과 해석 문구

예측 결과는 `st.metric`, `st.progress`, `st.success`, `st.warning`, `st.error` 등을 활용하여 직관적으로 확인할 수 있도록 구성했습니다.

### 실제값 vs 예측값 탭

테스트 데이터에 대한 실제값과 예측값을 비교할 수 있는 기능을 제공합니다.

주요 기능은 다음과 같습니다.

- 전체 데이터 수 표시
- 정확히 예측한 개수 표시
- 오분류 개수 표시
- 전체 / 정확 / 오분류 필터링
- 예측 결과 표 출력
- 결과 CSV 다운로드 기능 제공

이를 통해 모델이 어떤 데이터를 정확히 예측했고, 어떤 데이터를 오분류했는지 확인할 수 있도록 구성했습니다.

---

## 프로젝트 구조

```text
project/
├── app.py
├── model.ipynb
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
├── model/
│   ├── addiction_model.h5
│   └── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 실행 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/David35798/deeplearningProject01.git
cd deeplearningProject01
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 모델 파일 생성

`model/` 폴더가 없거나 `addiction_model.h5`, `scaler.pkl` 파일이 없는 경우 먼저 `model.ipynb`를 실행하여 모델과 scaler를 생성합니다.

생성되는 파일은 다음과 같습니다.

```text
model/addiction_model.h5
model/scaler.pkl
```

### 4. Streamlit 실행

```bash
streamlit run app.py
```

---

## 본 프로젝트를 통해 배운 점

- 딥러닝 모델을 활용하여 이진 분류 문제를 해결하는 과정을 경험했습니다.
- 스마트폰 사용 패턴 데이터를 기반으로 중독 여부를 예측하는 모델을 구현했습니다.
- `StandardScaler`를 활용하여 입력 데이터를 표준화하는 전처리 과정을 적용했습니다.
- Dense Layer와 Dropout을 활용하여 딥러닝 분류 모델을 설계했습니다.
- Sigmoid 출력값을 활용하여 예측 확률을 계산하고, 이를 위험도 단계로 변환하는 방식을 구현했습니다.
- 학습된 모델과 scaler를 저장한 뒤, Streamlit 웹 애플리케이션에서 재사용하는 전체 흐름을 구현했습니다.
- 실제값과 예측값을 비교하고, 정확 / 오분류 결과를 확인하는 기능을 구현했습니다.
- 데이터 전처리, 모델 학습, 성능 평가, 모델 저장, 웹 서비스 구현까지 딥러닝 프로젝트의 전체 파이프라인을 경험했습니다.

---

## 향후 개선 방향

- 추가 변수 활용
  - 나이
  - 성별
  - 주말 스마트폰 사용시간
  - 스트레스 수준
  - 학업/업무 영향 여부
- 다양한 모델과의 성능 비교
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Deep Neural Network 구조 변경
- 하이퍼파라미터 튜닝
  - 은닉층 개수
  - 뉴런 수
  - Dropout 비율
  - Learning Rate
  - Batch Size
- EarlyStopping 적용을 통한 과적합 방지
- 모델 성능 평가 지표 추가
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- 예측 결과에 대한 상세한 설명 기능 추가
- 사용자별 예측 기록 저장 기능 추가
- Streamlit Cloud 또는 Docker 기반 배포

---

## 프로젝트 상세 자료

프로젝트의 전체 설명 자료는 아래 링크에서 확인할 수 있습니다.

- [설명 자료 보기](https://docs.google.com/presentation/d/15lF_likldf6oBaduDcCsy9tWQT7OvYRJ/edit?usp=sharing&ouid=113197474989675182723&rtpof=true&sd=true)
