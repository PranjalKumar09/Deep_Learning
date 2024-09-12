"""  
    CREATE TABLE IF NOT EXISTS Location (
    location_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    city,
    state)

    CREATE TABLE IF NOT EXISTS Crime_Incident (
        incident_id SERIAL PRIMARY KEY,
        location_id INT,
        crime_type VARCHAR(20) CHECK (crime_type IN ('Rape', 'Kidnapping', 'Stalking', 'Dowry', 'Other')),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (location_id) REFERENCES Location(location_id)
    )
    
    
     CREATE TABLE Hotspot (
        hotspot_id SERIAL PRIMARY KEY,
        location_id INT,
        crime_type VARCHAR(20) CHECK (crime_type IN ('Rape', 'Kidnapping', 'Stalking', 'Other')),
        incident_count INT,
        hotspot_status VARCHAR(3) CHECK (hotspot_status IN ('Yes', 'No')),
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (location_id) REFERENCES Location(location_id)
        
        
        
        
        """
""" 





"""

import psycopg2
import pandas as pd

# Database connection parameters
conn_params = {
    "host": "localhost",
    "database": "sih",
    "user": "postgres",
    "password": "0907"
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()

# SQL command to create the Location table
create_table_query = """
CREATE TABLE IF NOT EXISTS Location (
    location_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    city VARCHAR(255),
    state VARCHAR(255)
);
"""

# Execute the table creation query
cursor.execute(create_table_query)
conn.commit()

# Read data from Excel file
file_path = 'locations_data.xlsx'
df = pd.read_excel(file_path)

# Insert data into the Location table
for index, row in df.iterrows():
    insert_query = """
    INSERT INTO Location (name, latitude, longitude, city, state) 
    VALUES (%s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, (row['address'], row['latitude'], row['longitude'], row['city'], row['state']))

# Commit the transaction
conn.commit()

# Close the database connection
cursor.close()
conn.close()

print("Data inserted successfully!")


