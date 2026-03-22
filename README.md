README File
-----------------------------------------------



PROJECT TITLE
-------------
Bank Marketing Campaign Prediction using MLOps


1. PROBLEM STATEMENT
-------------------
- Many Banks follow a brute force method of reaching out to every customer of theirs to offer their products like term-deposit.

- Reaching out to every one is not cost effective and sometimes not practical for large banks. Moreover even if that is done only a small percentage of customers actually subscribe to such products.

In view of the above points, we need to solve the problem by building a machine learning model by using industry standard ML Ops practices to assist the bank achieve its objective which is predicting which customers to target (or not target) so that there are higher chances of reaching out to the right set of people who would subscribe to the term-deposit.


2. DATASET DESCRIPTION
----------------------
- The dataset provided as part of this problem is a set of customers, their demographics, past cmapaign outcomes, few macro-economic indicators and whether the customer subscribed to the term-deposit or not.

-In short, the dataset is a snapshot that provides

	- Customer Demographics like age, job, marital status, education.
	- Past campaign details like how many times a customer was contacted, time since last contacted.
	- Economic indicators like euro ibor rate, consumer confidence, employment rate.
	- Target variable "y" that tells if a customer subscribed to the term deposit or not. It is either a yes or a no.


3. APPROACH AND MODEL
---------------------
- Model Used: Logistic Regression

Preprocessing Steps:
- Categorical variables handled via one-hot encoding
- Numerical features scaled via StandardScaler
- A feature that is only available after the call is made i.e. "duration" is removed

CITATION: "Duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y=’no’). Yet, the duration is not known before a call is performed. Also, after the end of the call y is known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model." (Reference: https://medium.com/%40mauridurcak/bussines-analitycs-and-logistic-regression-for-bank-marketing-dataset-1f20cb829c85)

The trained model and scaler saved as:
- 12510050_logistic_model.pkl
- 12510050_scaler.pkl


4. SYSTEM WORKFLOW
------------------
1. Customer data is accepted as a JSON
2. Flask API loads the trained model and scaler
3. Input data is preprocessed
4. Model predicts subscription outcome (0 as "no", 1 as "yes")
5. Prediction is returned via REST API


5. LOCAL EXECUTION
------------------
1. Navigate to project directory
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Flask application:
   python app.py
4. Access API at:
   http://localhost:5000


6. DOCKER CONTAINERIZATION
--------------------------
Docker has been used to package the application and its dependencies.

Build Docker Image:
docker build -t 12510050-mlops-model .

Run Docker Container:
docker run -p 5000:5000 12510050-mlops-model

The API becomes accessible at:
http://localhost:5000


7. AWS CLOUD DEPLOYMENT
----------------------
The Dockerized application is deployed on an AWS EC2 instance.

Deployment Steps:

- Launch EC2 instance (Amazon Linux 2/AMI, t3.micro)
- Open inbound port 5000 in security group and define the port access as Custom TCP
- Install Docker on EC2
- Copy project files to EC2 using SCP
- Build and run Docker container on EC2

The API is accessible via the public IP of the EC2 instance.


8. API USAGE (INFERENCE)
------------------------
Endpoint:
POST /predict

Sample Requests (JSON):
{
  "age": 42,
  "campaign": 1,
  "pDays": 5,
  "previous": 1,
  "pDaysFlag": 1,
  "consPriceIdx": 92.8,
  "consConfIdx": -30.0,
  "eurIbor3M": 1.2,
  "nrEmployed": 5000.0
}

Sample Response:
{"prediction": 1}

{
  "age": 37,
  "campaign": 1,
  "pDays": 999,
  "previous": 0,
  "pDaysFlag": 0,
  "consPriceIdx": 93.2,
  "consConfIdx": -40.0,
  "eurIbor3M": 4.85,
  "nrEmployed": 5191.0
}

Sample Response:
{"prediction": 1}

9. PROJECT STRUCTURE
-------------------
12510050_mlops_model/
|-- app.py
|-- Dockerfile
|-- requirements.txt
|-- 12510050_logistic_model.pkl
|-- 12510050_scaler.pkl
|-- README.txt


10. RESULTS AND CONCLUSION
--------------------------
The project shows a full MLOps pipeline, including model development, 
API creation, containerization, and cloud deployment. 

The deployed system help provides scalable prediction that is useful because 
the bank can use the recommendation to understand whom to not target and save costs 
in similar future campaigns.



---------------------------END------------------------------------

MERRY CHRISTMAS, SEASON'S GREETINGS AND HAPPY NEW YEAR 2026!!!
