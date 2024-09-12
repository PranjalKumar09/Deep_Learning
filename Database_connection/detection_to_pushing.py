import requests

def get_coordinates_from_ip(ip_address):
    try:
        response = requests.get(f"https://ipapi.co/{ip_address}/json/")
        data = response.json()
        latitude = data.get('latitude', None)
        longitude = data.get('longitude', None)
        return latitude, longitude
    except Exception as e:
        print(f"Error retrieving coordinates: {e}")
        return None, None

# Example usage
ip_address = "your_camera_ip"
latitude, longitude = get_coordinates_from_ip(ip_address)
print(f"Latitude: {latitude}, Longitude: {longitude}")
