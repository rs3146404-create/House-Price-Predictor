import os
from pyexpat import features
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from config import views
# import json
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'house_rent_model.pkl')


def get_base_context():
    """Return base context data for all views"""
    return {
        'title': 'House Rent Predictor',
        'description': 'Predict monthly house rent instantly using Machine Learning.',
        'app_version': '1.0',
        'floor_options': ['1 out of 2','Ground out of 2','2 out of 3','2 out of 4','1 out of 3','3 out of 4','Ground out of 3'],
        'area_types':['Super Area','Carpet Area','Built Area'],
        'cities': ['Mumbai','Chennai','Bangalore','Hyderabad','Delhi','Kolkata'],
        'furnishing_types': ['Furnished', 'Unfurnished', 'Semi-Furnished'],
        'tenant_types': ['Bachelors/Family', 'Bachelors', 'Family'],
    }

def index(request):
    context = get_base_context()
    return render(request, 'index.html', context)

def calculate_rent(input_data):
    """Simple ML-like calculation for rent prediction"""
    input = input_data
    model = joblib.load(model_path)

    prediction = model.predict(input)
    original = np.expm1(prediction)

    return int(original[0])

@csrf_exempt
def predict(request):
    if request.method == 'POST':

        
        try:
            # Get form data
            bhk = int(request.POST.get('bhk', 1))
            size = int(request.POST.get('size', 1000))
            floor = request.POST.get('floor', 'Floor')
            area_type = request.POST.get('area_type', 'Area Type')
            furnishing = request.POST.get('Furnishing Status', 'Furnishing Status')
            tenant = request.POST.get('Tenant Preferred', 'Tenant Preferred')
            city = request.POST.get('city', 'City')
            bathroom = int(request.POST.get('bathroom', 1))

            # dataframe
            input_data = pd.DataFrame({
                'BHK': [bhk],
                'Size': [size],
                'Floor': [floor],
                'Area Type': [area_type],
                'Furnishing Status': [furnishing],
                'Tenant Preferred': [tenant],
                'City': [city],
                'Bathroom': [bathroom]
            })

            # Calculate prediction
            prediction = calculate_rent(input_data)
            
            # Prepare context with prediction
            context = get_base_context()
            context['prediction'] = prediction
            context['show_result'] = True
            context['inputs'] = {
                'bhk': bhk,
                'size': size,
                'floor': floor,
                'furnishing': furnishing,
                'tenant': tenant,
                'city': city,
                'area_type': area_type,
                'bathroom': bathroom
            }
            
            return render(request, 'index.html', context)
        except Exception as e:
            context = get_base_context()
            context['error'] = str(e)
            return render(request, 'index.html', context)
    
    return index(request)

