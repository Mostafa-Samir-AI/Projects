# **Customer Segmentation Project**

## **1. Overview**

This project performs **customer segmentation** using a combination of unsupervised learning techniques, primarily **K-Means clustering**, together with a production-ready deployment pipeline. The solution includes:

* Data ingestion and cleaning
* Exploratory Data Analysis (EDA)
* Feature preprocessing
* Model training using K-Means
* Saving scalers and models
* Deployment using **FastAPI**, **Docker**, and **docker-compose**
* Experiment tracking via **MLflow**

The objective is to deliver a reproducible, modular, and deployable machine learning system for segmenting customers based on their characteristics and spending behavior.

---

## **2. Project Structure**

Based on your current directory:

```
Customer_Segmentation_Project/
│
├── Mostafa Samir.pptx
├── README.md
│
├── data/
│   ├── customer_segmentation_data.csv
│   ├── kmeans_data.csv
│   ├── k_pro_data.csv
│   ├── test_Data.csv
│   ├── visual_data.csv
│   └── Segmentation data legend.xlsx
│
├── Deployment/
│   ├── deploy.py
│   ├── docker-compose.yaml
│   ├── dockerfile
│   ├── kmeans_model.pkl
│   ├── mlflow.db
│   ├── request.py
│   ├── requirements.txt
│   └── scale.pkl
│
├── mlruns/
│   ├── 1/
│   └── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── k_means.ipynb
│   ├── k_prototype.ipynb
│   ├── Deploy_test.ipynb
│   └── test.py
│
└── __pycache__/
    └── MyDeploy.cpython-312.pyc
```

---

## **3. Data Pipeline**

### **3.1 Data Ingestion**

The raw datasets under `data/` are loaded and checked for:

* Missing values
* Duplicates
* Incorrect data types
* Outliers (optional)

The ingestion step ensures the dataset is clean and consistent before analysis.

### **3.2 Data Preparation & Cleaning**

Performed mainly in notebooks such as **EDA.ipynb**, involving:

* Handling missing values
* Removing inconsistent entries
* Normalization and scaling (using `scale.pkl`)
* Selecting relevant features for clustering

Visualizations of distributions and correlations are stored in `visual_data.csv` and notebook outputs.

---

## **4. Modeling**

### **4.1 K-Means Clustering**

The primary model used is **K-Means**, trained using the processed dataset.
Steps include:

* Determining optimal **k** (Elbow Method / Silhouette Score)
* Fitting the model on scaled data
* Evaluating cluster separation
* Labeling clusters with descriptive interpretations

The final model is stored as:

```
Deployment/kmeans_model.pkl
```

The scaler used during preprocessing is stored as:

```
Deployment/scale.pkl
```

### **4.2 K-Prototypes (Optional)**

The notebook **k_prototype.ipynb** demonstrates segmentation with mixed numerical + categorical features (if applicable).

---

## **5. Deployment**

### **5.1 FastAPI Server (`deploy.py`)**

The deployment script:

* Loads `kmeans_model.pkl` and `scale.pkl`
* Accepts JSON requests via POST
* Preprocesses user input
* Returns the predicted cluster ID

### **5.2 Request Example (`request.py`)**

A small client script inside `Deployment/` demonstrating how to send a request to the API.

### **5.3 Docker Deployment**

You can build and run the application in a container:

```bash
docker build -t customer-segmentation -f Deployment/dockerfile .
docker run -p 8000:8000 customer-segmentation
```

Or use docker-compose:

```bash
docker-compose -f Deployment/docker-compose.yaml up --build
```

This ensures reproducibility and isolated environments for production.

---

## **6. Experiment Tracking**

The folder:

```
mlruns/
```

contains MLflow tracking data, including:

* Model parameters
* Metrics
* Artifacts
* Experiments

The local MLflow database is:

```
Deployment/mlflow.db
```

To view MLflow UI:

```bash
mlflow ui --backend-store-uri Deployment/mlflow.db
```

---

## **7. Notebooks Overview**

### **EDA.ipynb**

Complete exploratory analysis: distributions, correlations, and feature relationships.

### **k_means.ipynb**

The core notebook where:

* Data is prepared
* K-Means is trained
* Metrics and visualizations are produced
* Model and scaler are saved

### **Deploy_test.ipynb**

Tests the FastAPI deployment endpoint interactively.

### **k_prototype.ipynb**

Alternative clustering technique for mixed data types.

---

## **8. Requirements**

The full list of Python dependencies is stored in:

```
Deployment/requirements.txt
```

Typical packages include:

* pandas
* numpy
* scikit-learn
* fastapi
* uvicorn
* mlflow
* pydantic
* python-multipart

Install dependencies:

```bash
pip install -r Deployment/requirements.txt
```

---

## **9. Usage**

### **Start the API locally**

```bash
python Deployment/deploy.py
```

### **Send a prediction request**

```bash
python Deployment/request.py
```

### **Run inside Docker**

```bash
docker-compose up --build
```

---

## **10. Future Enhancements**

* Add PCA visualization endpoints in FastAPI
* Add a UI dashboard (Streamlit / Gradio)
* Deploy to cloud (AWS/GCP/Render)
* Expand to deep clustering models
* Automate ML pipeline with Prefect or Airflow

