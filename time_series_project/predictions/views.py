import json
import os
import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import ModelTester 

# path to the folder containing your models
MODEL_FOLDER = "models"

@csrf_exempt 
def predict(request):
    if request.method == 'POST':
        try:
            # Parse the incoming JSON data
            data = json.loads(request.body)
            
            # Get dataset ID and values from the request body
            dataset_id = data['dataset_id']
            values = data['values']

            # Create a ModelTester instance
            model_tester = ModelTester(models_folder=MODEL_FOLDER)

            # Pass the dataset_id and values to the test_model method
            prediction_result = model_tester.test_model({
                'dataset_id': dataset_id,
                'values': values
            })
            
            return JsonResponse(prediction_result)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method, only POST allowed."}, status=400)
