from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import pickle
import json
import os
from django.conf import settings


# Load the trained model
model_path = os.path.join(settings.BASE_DIR, 'liver_disease_prediction', 'Liver_trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)  # Parse JSON data from the request body
            extracted_data = data.get('features', [])  # Extract features from the JSON

            features = tuple(extracted_data)
            print(features)

            if not features or len(features) != 11:  # Assuming 8 features
                return JsonResponse({'error': 'Invalid input data or incomplete'}, status=400)

            input_data_as_numpy_array = np.asarray(features)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            prediction = loaded_model.predict(input_data_reshaped)
            print(f"Prediction: {prediction}")

            if (prediction[0] == 0):
                message = 'The Person does not have a Liver Disease'
            else:
                message = 'The Person has Liver Disease'

            return JsonResponse({'message': message})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)