from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def get_location_info_by_coordinates(lat, lon, timeout=10):
    geolocator = Nominatim(user_agent="my_app", timeout=timeout)

    try:
        # Reverse geocode the coordinates
        location = geolocator.reverse((lat, lon), language='en', exactly_one=True)
        if location:
            address_info = location.raw.get('address', {})

            # Extract state, city, pincode, and full address
            state = address_info.get('state', 'State not found')
            city = address_info.get('city', address_info.get('town', address_info.get('village', 'City not found')))
            pincode = address_info.get('postcode', 'Pincode not found')
            full_address = location.address

            return {
                "state": state,
                "city": city,
                "pincode": pincode,
                "full_address": full_address,
                "latitude": location.latitude,
                "longitude": location.longitude
            }
        else:
            return None

    except GeocoderTimedOut:
        print("Geocoding service timed out. Please try again.")
        return None

# Example usage with coordinates for central India
latitude = 20.5937
longitude = 78.9629

location_info = get_location_info_by_coordinates(latitude, longitude)
if location_info:
    print(f"Full Address: {location_info['full_address']}")
    print(f"State: {location_info['state']}")
    print(f"City: {location_info['city']}")
    print(f"Pincode: {location_info['pincode']}")
    print(f"Latitude: {location_info['latitude']}")
    print(f"Longitude: {location_info['longitude']}")


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def get_location_info_by_address(address, timeout=10):
    geolocator = Nominatim(user_agent="my_app", timeout=timeout)

    try:
        # Forward geocode the address
        location = geolocator.geocode(address)
        if location:
            address_info = location.raw.get('address', {})

            # Extract state, city, pincode, and full address
            state = address_info.get('state', 'State not found')
            city = address_info.get('city', address_info.get('town', address_info.get('village', 'City not found')))
            pincode = address_info.get('postcode', 'Pincode not found')
            full_address = location.address

            return {
                "state": state,
                "city": city,
                "pincode": pincode,
                "full_address": full_address,
                "latitude": location.latitude,
                "longitude": location.longitude
            }
        else:
            print("Address not found")
            return None

    except GeocoderTimedOut:
        print("Geocoding service timed out. Please try again.")
        return None

# Example usage with an address
address = "175 5th Avenue, NYC"

location_info = get_location_info_by_address(address)
if location_info:
    print(f"Full Address: {location_info['full_address']}")
    print(f"State: {location_info['state']}")
    print(f"City: {location_info['city']}")
    print(f"Pincode: {location_info['pincode']}")
    print(f"Latitude: {location_info['latitude']}")
    print(f"Longitude: {location_info['longitude']}")
""" 
India Gate,  New Delhi
21-12-2015 15:26:36
New Delhi
Delhi
"""




"""

Now i have states as 

Maharashtra
Delhi
Karnataka
Rajasthan
Tamil Nadu
West Bengal
Telangana
Gujarat
Uttar Pradesh
Madhya Pradesh
Punjab
Haryana
Andhra Pradesh
Jharkhand
Chhattisgarh
Himachal Pradesh
Uttarakhand
Jammu & Kashmir
Odisha
Tripura
Nagaland
Manipur
Sikkim
Meghalaya
Mizoram


but in  
india_map = gpd.read_file('Untitled Folder/STATE_BOUNDARY.shp')


print("Columns in india_map:")
print(india_map.columns)
print(india_map['STATE'])


Columns in india_map:
Index(['STATE', 'State_LGD', 'Shape_Leng', 'Shape_Area', 'geometry'], dtype='object')
0                              ANDAMAN & NICOBAR
1                                 ANDHRA PRADESH
2                              ARUN>CHAL PRADESH
3                                          ASSAM
4                                          BIH>R
5                                     CHAND|GARH
6                                   CHHAtT|SGARH
7             D>DRA & NAGAR HAVELI & DAM>N & DIU
8                                          DELHI
9            DISPUTED (MADHYA PRADESH & GUJAR>T)
10         DISPUTED (MADHYA PRADESH & R>JASTH>N)
11                 DISPUTED (R>JATH>N & GUJAR>T)
12    DISPUTED (WEST BENGAL , BIH>R & JH>RKHAND)
13                                           GOA
14                                       GUJAR>T
15                                       HARY>NA
16                              HIM>CHAL PRADESH
17                             JAMMU AND KASHM|R
18                                     JH>RKHAND
19                                     KARN>TAKA
20                                        KERALA
21                                        LAD>KH
22                                   LAKSHADWEEP
23                                MADHYA PRADESH
24                                   MAH>R>SHTRA
25                                       MANIPUR
26                                     MEGH>LAYA
27                                       MIZORAM
28                                      N>G>LAND
29                                        ODISHA
30                                    PUDUCHERRY
31                                        PUNJAB
32                                     R>JASTH>N
33                                        SIKKIM
34                                    TAMIL N>DU
35                                     TELANG>NA
36                                       TRIPURA
37                                   UTTAR>KHAND
38                                 UTTAR PRADESH
39                                   WEST BENGAL




Location (
    location_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    city VARCHAR(255),
    state VARCHAR(255)
    
    
    
 Crime_Incident (
    incident_id SERIAL PRIMARY KEY,
    location_id INT,
    crime_type VARCHAR(20) CHECK (crime_type IN ('Rape', 'Kidnapping', 'Dowry', 'Stalking', 'Other')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (location_id) REFERENCES Location(location_id)
)


INSERT INTO Hotspot (location_id, crime_type, incident_count, hotspot_status)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (location_id, crime_type)
    DO UPDATE SET
        incident_count = EXCLUDED.incident_count,
        hotspot_status = EXCLUDED.hotspot_status,
        last_updated = CURRENT_TIMESTAMP;

Location (
    location_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    city VARCHAR(255),
    state VARCHAR(255)
    
    
    
 Crime_Incident (
    incident_id SERIAL PRIMARY KEY,
    location_id INT,
    crime_type VARCHAR(20) CHECK (crime_type IN ('Rape', 'Kidnapping', 'Dowry', 'Stalking', 'Other')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (location_id) REFERENCES Location(location_id)
)


INSERT INTO Hotspot (location_id, crime_type, incident_count, hotspot_status)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (location_id, crime_type)
    DO UPDATE SET
        incident_count = EXCLUDED.incident_count,
        hotspot_status = EXCLUDED.hotspot_status,
        last_updated = CURRENT_TIMESTAMP;




"""


"""
diffeerent  states comparsion on india map like on different crime types    ,  plotting different states
















"""






""" 

"""