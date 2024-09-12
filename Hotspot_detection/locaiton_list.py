from collections import defaultdict, Counter



""" 


Python list with 20 addresses in "CITY" including crime locations, public streets, and bypasses. Each entry contains the address, latitude, longitude (with 6-digit precision),"city" and state.
like 
[
    {
        "address": "---,
        "latitude": XX.XXXXXX,
        "longitude":  XX.XXXXXX,
        "city": "---",
        "state": "----"
    },
    {--}---

"""

""""city"_ difference -> Meerut Sehore
TOTAL LOCATION->  976

STATE(Cities_Count) ->  Citie1(Location1), Citie2(Location2) ----

Maharashtra(6):         Mumbai(20), Pune(21), Nagpur(20), Nashik(21), Kalyan(21), Aurangabad(22)
Delhi(2):               New Delhi(15), Delhi(5)
Karnataka(2):           Bengaluru(20), Mysore(21)
Rajasthan(3):           Jaipur(20), Jodhpur(20), Bikaner(20)
Tamil Nadu(2):          Chennai(21), Coimbatore(21)
West Bengal(2):         Kolkata(20), Siliguri(21)
Telangana(1):           Hyderabad(20)
Gujarat(4):             Ahmedabad(21), Surat(21), Vadodara(21), Rajkot(20)
Uttar Pradesh(4):       Lucknow(21), Ghaziabad(20), Kanpur(21), Agra(21)
Madhya Pradesh(4):      Bhopal(21), Indore(21), Jabalpur(20), Gwalior(21)
Punjab(2):              Ludhiana(21), Amritsar(20)
Haryana(1):             Faridabad(21)
Andhra Pradesh(1):      Vijayawada(21)
Jharkhand(1):           Ranchi(21)
AP(1):                  Kakinada(21)
Chhattisgarh(1):        Raipur(21)
Himachal Pradesh(1):    Shimla(21)
Uttarakhand(1):         Dehradun(21)
Jammu & Kashmir(1):     Jammu(21)
Odisha(2):              Bhubaneswar(20), Rourkela(22)
Tripura(1):             Agartala(21)
Nagaland(1):            Kohima(21)
Manipur(1):             Imphal(21)
Sikkim(1):              Gangtok(21)
Meghalaya(1):           Shillong(21)
Mizoram(1):             Aizawl(22)


Crime Types : 'Rape', 'Kidnapping', 'Stalking', 'Dowry','Other'

"""

"""


BIASNESS
==========

Rajasthan, Uttar Pradesh, Madhya Pradesh, Maharashtra, Kerala, Assam, Haryana, Jharkhand, Odisha, and Delhi -> Rapes





Nagaland -> Safest


Dowry related-> UP (2302), Bihar(1047), Madhya Pradesh (627), Rajasthan (480) , Bengal (523), Odhisha(320), Jharkhand(280), Haryana (480)
Except Assam all north east compelte dowry free
Total -> Delhi or New Delhi, Surat, Kochi, Ahmedabad, Chennai, Jaipur, Patna,  Ghaziabad, Mumbai

Stalking -> UP 2069, Delhi 422, Mp 242, Maharasthra 231 , Haryana 199


Overall -> Uttar Pradesh (56,083), Rajasthan (40,738), and Maharashtra (39,526), whereas the UTs of Ladakh (18) and Lakshadweep (9) recorded the lowest.









"""


""" 
Threshold
2. Methods for Setting Hotspot Thresholds

a. Quantile-Based Thresholds

    Top Percentiles:
        High Threshold: Identify the top 5% or 10% of locations with the highest number of incidents. This method is useful for pinpointing the most problematic areas.
        Example: If you have 1,020 locations, the top 10% would be approximately the top 102 locations.

    Bottom Percentiles:
        Low Threshold: Identify the bottom 10% to understand areas with minimal crime.

b. Mean and Standard Deviation

    Mean + 1 Standard Deviation:
        Calculate the average number of incidents across all locations and add one standard deviation to set a threshold. This will help in identifying locations with significantly more incidents than average.
        Example Calculation:
            Mean incidents per location = 10
            Standard deviation = 5
            Threshold = Mean + 1 SD = 10 + 5 = 15 incidents

    Mean + 2 Standard Deviations:
        For a stricter threshold, you can use the mean plus two standard deviations. This will highlight locations with extremely high incident rates.
        Example Calculation:
            Threshold = Mean + 2 SD = 10 + 2(5) = 20 incidents


10240


"3-13"

"""

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
