# Django Model Prediction API

This project provides a Django-based API that allows users to get predictions from a machine learning model. It supports loading a model (saved using `joblib`), preprocessing time-series data, and returning predictions based on the provided input.

## Features
- Django REST API for model prediction.
- Time-series data preprocessing with lags and rolling window features.
- Support for multiple machine learning models (trained and saved using `joblib`).
- Containerized using Docker.

## Prerequisites

- Docker (for containerization)

## Installation

To run this project locally or using Docker, follow the steps below.

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yaramostafa/Timeseries-Forecasting
cd D:/Timeseries-Forecasting/django-model-prediction
```

Or just open this Dir and download it: https://github.com/yaramostafa/Timeseries-Forecasting/tree/c0f0b782abbf2694848c95fed19362f845055541/time_series_project

### 2. Running the Project with Docker
- Build the Docker Image
Open wsl (in the project directory where the docker-file is)

Run
```
docker build -t django-model-prediction .
```
- Run the Docker Container
After the image is built, run the Docker container:
```
docker run -p 8000:8000 django-model-prediction
```
This will expose the API on: ```http://localhost:8000/api/predict/```
---

### 3. Testing using postman 
Use ```http://localhost:8000/api/predict/```
**Request Body:**
```
{
  "dataset_id": "test_102",
  "values": [
    {
      "time": "2021-11-25T00:00:00",
      "value": -0.9753684650148648
    },
    {
      "time": "2021-11-26T00:00:00",
      "value": -0.908480414750474
    },
    {
      "time": "2021-11-27T00:00:00",
      "value": -0.8152620300016913
    },
    {
      "time": "2021-11-28T00:00:00",
      "value": -1.0885601661132862
    },
    {
      "time": "2021-11-29T00:00:00",
      "value": -0.8796273354358487
    },
    {
      "time": "2021-11-30T00:00:00",
      "value": -1.0598928922740152
    },
    {
      "time": "2021-12-01T00:00:00",
      "value": null
    },
    {
      "time": "2021-12-02T00:00:00",
      "value": -1.0572877628128294
    },
    {
      "time": "2021-12-03T00:00:00",
      "value": -0.931850290323409
    },
    {
      "time": "2021-12-04T00:00:00",
      "value": -0.9066872074281134
    },
    {
      "time": "2021-12-05T00:00:00",
      "value": -1.0784178357532677
    },
    {
      "time": "2021-12-06T00:00:00",
      "value": -1.1433275313846551
    },
    {
      "time": "2021-12-07T00:00:00",
      "value": null
    },
    {
      "time": "2021-12-08T00:00:00",
      "value": -1.2037721354150372
    },
    {
      "time": "2021-12-09T00:00:00",
      "value": -1.080252403183123
    },
    {
      "time": "2021-12-10T00:00:00",
      "value": -0.973488778015679
    },
    {
      "time": "2021-12-11T00:00:00",
      "value": -1.0306239182965182
    },
    {
      "time": "2021-12-12T00:00:00",
      "value": -0.9521110752943764
    },
    {
      "time": "2021-12-13T00:00:00",
      "value": -0.8537815660276247
    },
    {
      "time": "2021-12-14T00:00:00",
      "value": -0.95652540313693
    },
    {
      "time": "2021-12-15T00:00:00",
      "value": -0.8790172195106937
    },
    {
      "time": "2021-12-16T00:00:00",
      "value": -0.902792921399683
    },
    {
      "time": "2021-12-17T00:00:00",
      "value": -0.8684325492476539
    },
    {
      "time": "2021-12-18T00:00:00",
      "value": -0.9621477215581422
    },
    {
      "time": "2021-12-19T00:00:00",
      "value": -1.0013616308239526
    },
    {
      "time": "2021-12-20T00:00:00",
      "value": -0.9378574021420448
    },
    {
      "time": "2021-12-21T00:00:00",
      "value": -1.0677658017153189
    },
    {
      "time": "2021-12-22T00:00:00",
      "value": -0.9679282038060564
    },
    {
      "time": "2021-12-23T00:00:00",
      "value": -0.9835756729730298
    },
    {
      "time": "2021-12-24T00:00:00",
      "value": -0.9100977067999184
    },
    {
      "time": "2021-12-25T00:00:00",
      "value": -1.1450151079230508
    },
    {
      "time": "2021-12-26T00:00:00",
      "value": -1.1019343750893495
    },
    {
      "time": "2021-12-27T00:00:00",
      "value": -0.9050772336086048
    }
    
  ]
}
```
**Response Body:**
```
{
    "prediction": -0.9622776369622751
}
```
