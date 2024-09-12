""", Mumbai , Pune , Nagpur , Nashik , Kalyan , Aurangabad , New Delhi , Delhi , Bengaluru , Mysore , Jaipur , Jodhpur , Bikaner , Chennai , Coimbatore , Kolkata , Siliguri , Hyderabad , Ahmedabad , Surat , Vadodara , Rajkot , Lucknow , Ghaziabad , Kanpur , Agra , Bhopal , Indore , Jabalpur , Gwalior , Ludhiana , Amritsar , Faridabad , Vijayawada , Ranchi , Kakinada , Raipur , Shimla , Dehradun , Jammu , Bhubaneswar , Rourkela , Agartala , Kohima , Imphal , Gangtok , Shillong , Aizawl"""

import random
import pandas as pd
from datetime import datetime, timedelta

# Crime type bias
crime_type_bias = {
    'Rape': {
        'Maharashtra': 0.081,
        'Delhi': 0.045,
        'Karnataka': 0.018,
        'Rajasthan': 0.214,
        'Tamil Nadu': 0.013,
        'West Bengal': 0.038,
        'Telangana': 0.022,
        'Gujarat': 0.019,
        'Uttar Pradesh': 0.109,
        'Madhya Pradesh': 0.089,
        'Punjab': 0.036,
        'Haryana': 0.053,
        'Andhra Pradesh': 0.039,
        'Jharkhand': 0.05,
        'Chhattisgarh': 0.037,
        'Himachal Pradesh': 0.013,
        'Uttarakhand': 0.019,
        'Jammu & Kashmir': 0.008,
        'Odisha': 0.049,
        'Tripura': 0.003,
        'Nagaland': 0.0,
        'Manipur': 0.001,
        'Sikkim': 0.0,
        'Meghalaya': 0.004,
        'Mizoram': 0.001
    },
    'Kidnapping': {
        'Maharashtra': 0.150,
        'Delhi': 0.113,
        'Karnataka': 0.041,
        'Rajasthan': 0.136,
        'Tamil Nadu': 0.039,
        'West Bengal': 0.128,
        'Telangana': 0.054,
        'Gujarat': 0.080,
        'Uttar Pradesh': 0.195,
        'Madhya Pradesh': 0.098,
        'Punjab': 0.046,
        'Haryana': 0.073,
        'Andhra Pradesh': 0.065,
        'Jharkhand': 0.061,
        'Chhattisgarh': 0.026,
        'Himachal Pradesh': 0.021,
        'Uttarakhand': 0.017,
        'Jammu & Kashmir': 0.014,
        'Odisha': 0.032,
        'Tripura': 0.009,
        'Nagaland': 0.006,
        'Manipur': 0.005,
        'Sikkim': 0.004,
        'Meghalaya': 0.003,
        'Mizoram': 0.002
    },
    'Dowry': {
        'Uttar Pradesh': 0.34,
        'Bihar': 0.14,
        'Madhya Pradesh': 0.062,
        'Rajasthan': 0.048,
        'West Bengal': 0.052,
        'Haryana': 0.048
    },
    'Stalking': {
        'Uttar Pradesh': 0.2069,
        'Delhi': 0.0482,
    }
}

# Ranking for bias
state_crime_ranking = ["Uttar Pradesh", "Rajasthan", "Maharashtra", "West Bengal", "Madhya Pradesh"]
city_ranking = ["New Delhi", "Surat", "Kochi", "Chennai", "Jaipur", "Patna", "Ghaziabad", "Mumbai"]

# Function to generate incidents
def generate_incidents(total_incidents, location_data):
    incidents = []
    
    for i in range(total_incidents):
        # Choose crime type with bias
        crime_type = random.choices(list(crime_type_bias.keys()))[0]

        # Choose a state with bias
        state_weights = list(crime_type_bias[crime_type].values())
        state_choices = list(crime_type_bias[crime_type].keys())
        state = random.choices(state_choices, weights=state_weights)[0]

        # Choose a city from that state (random if no city ranking)
        possible_cities = [loc for loc in location_data if loc['state'] == state]
        if possible_cities:
            if any(city in city_ranking for city in [c['city'] for c in possible_cities]):
                city = random.choice([c for c in possible_cities if c['city'] in city_ranking])
            else:
                city = random.choice(possible_cities)
        else:
            continue
        
        # Generate random timestamp
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))
        
        # Append incident
        incidents.append({
            'location_id': city['location_id'],
            'crime_type': crime_type,
            'timestamp': timestamp
        })
    
    return incidents
\\