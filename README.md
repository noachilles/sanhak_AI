# sanhak_AI
산학재단AI인턴십 4기/보험사 대상 보험 사기 가능성 예측 모델

🤝**팀 구성**: 5인팀  
👩‍💻**역할**: 머신러닝(Machine Learning) 모델 학습 및 데이터 전처리, 인코딩  
🧰**stacks**: ```python3``` ```sci-kit learn``` ```Flask``` ```RestfulAPI``` ```React```  

## 기획의도
* 교통사고에 대해 보험사 대상 보험 사기 가능성 예측 모델을 구현하고, 비교 분석해 최적화 및 웹 사이트 제작으로 서비스 제공  

## 프로젝트 진행 과정
### 활용 데이터
* 미국의 교통 사고 보험 청구 데이터  
✅데이터 확인하기 [Kaggle Data](https://www.kaggle.com/datasets/mykeysid10/insurance-claims-fraud-detection/data)  

### [데이터 전처리](https://github.com/noachilles/sanhak_AI/blob/main/sanhak_ai.ipynb)

**학습 변수 결정**
* 총 39개의 변수 중 필요하지 않다고 생각되는 변수는 회의를 통해 임의 결정하고, (fig 1. fig 2.)과 같이 시각화로 교차검증해 비율 확인 후 확정  

<div align="center"><img src="https://github.com/user-attachments/assets/796c7036-6ab3-42ff-aa0f-1c8f34aa3809" width=90%/></div>  
<div align="center">fig 1. Policy State-Fraud Reported(target)</div>
<div align="center"><img src="https://github.com/user-attachments/assets/bb50aa5e-ca53-4d6e-9a16-74bfaf025f93" width=90%/></div>  
<div align="center">fig 2. Policy State-Fraud Reported(Y/N)</div>


**결측값 처리**
* 1000개의 데이터 중 10% 미만의 비율이 NA 값을 가져 삭제 처리  

**파생 변수 생성**
* 피보험자의 보험 가입일과 사건 발생일 간의 차를 '**month_difference**'의 새로운 변수로 생성해 학습에 활용  

**이상치 분석**  
* 기초통계량 분석·확인 결과, 눈에 띄는 이상치 존재하지 않음  
* boxplot을 통해 이상치 조사 결과, 눈에 띄는 이상치 존재하지 않음  
<div align="center"><img src="https://github.com/user-attachments/assets/4a7220b4-8ba6-4a2d-84a5-3d4749be2afa"></div>  
<div align="center">fig 3. Box Plot</div>

**범주형 변수 인코딩**
* 종류에 따라 Ordinal Encoding/Binary Encoding을 적용  

**데이터 오버 샘플링**
* Data Imbalance가 발견됨  
<div align="center"><img src="https://github.com/user-attachments/assets/646756b5-b92c-4088-9267-67cd1e6e9c7d"/></div>  
<div align="center">fig 4. Target Data 값에 따른 개수</div>  

* 불균형 해소를 위해 데이터 오버 샘플링 알고리즘(SMOTE)를 적용  
* 기존 데이터셋으로 학습한 결과와 비교했을 때 오버 샘플링 후 학습된 결과가 향상됨   

<div align="center"><img src="https://github.com/user-attachments/assets/d3f04b3a-c57b-4fb8-942f-2f87873ad2db" width=80%/></div>  
<div align="center">fig 5. SMOTE 적용 전 학습 결과</div>
<div align="center"><img src="https://github.com/user-attachments/assets/499d25bc-8501-4efb-92d5-cab5dde03bd9" width=80%></div>  
<div align="center">fig 6. SMOTE 적용 후 학습 결과</div>


### MODEL  
**모델 학습**  
* 동일한 데이터셋을 7개의 모델에 대해 학습시킴  
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. Adaboost Classifier
    5. XGBoost Classifier
    6. LGBM Classifier
    7. CATBoost Classifier
* Overfitting을 방지하기 위해 Early Stopping 사용 후 Overfitting 재차 확인  

<div align="center"><img src="https://github.com/user-attachments/assets/f244b184-dab7-47a6-bf27-21d31120a6a6" width=90%/></div>  

<div align="center">fig 7. XGBoost Early Stopping Learning Curve</div>   


* 5개의 폴드로 나누어 교차 검증을 수행함 => AUC 확인  
* 각 모델에 대한 변수 중요도(feature importance)를 확인함  

**최종 선택 모델**  
* 모델 Test 결과를 바탕으로 가장 우수한 성능을 보인 XGBoost를 선택해 활용하고자 함  
* XGBoost와 CATBoost의 Feature Importance를 기준으로 13개의 변수(기존 39개)를 사용하는 것으로 확정함   
<b>** 보험사 입장에서 주어진 형식에 맞추어 모든 데이터를 입력해야 하므로 활용되는 attribute개수를 줄이고자 함</b> 

<div align="center"><img src="https://github.com/user-attachments/assets/f8c8ddb3-be49-4544-a577-73682caf0864" width=50%/></div>
<div align="center">fig 8. 최종 활용한 변수 리스트</div>

### 웹 개발  
<b>dir: project_api</b>  

**기능**
* 보험사 대상으로 접수된 보험 청구 데이터가 보험 사기일 가능성을 예측하는 서비스 제공  
* <b>입력 폼</b>: 확정된 13개의 변수를 format에 맞게 입력할 수 있는 폼 제공  
* <b>결과 페이지</b>: React Chart 라이브러리를 사용해 보험 사기 가능성을 하프도넛 그래프로 제공  
<div align="center"><img src="https://github.com/user-attachments/assets/2a5215d0-3744-4297-8cca-abd5b7d038fa" width=90%/></div>  
<div align="center">fig 9. 웹페이지 입력 폼(form) 예시 화면</div>
<div align="center"><img src="https://github.com/user-attachments/assets/d2a22f07-5aa3-4544-b247-194d7639e828" width=90%> </div>
<div align="center">fig 10. 웹페이지 예측 결과 예시 화면</div>