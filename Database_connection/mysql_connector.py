import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Admin@123",
            database="sih"
        )
        if db.is_connected():
            print("Successfully connected to MySQL database")
        return db
    except Error as e:
        print(f"Error: {e}")
        return None

def create_tables(db):
    try:
        cursor = db.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Location (
                            location_id INT AUTO_INCREMENT PRIMARY KEY,
                            name VARCHAR(255),
                            latitude FLOAT,
                            longitude FLOAT
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Person (
                            person_id INT AUTO_INCREMENT PRIMARY KEY,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            gender ENUM('Male', 'Female'),
                            location_id INT,
                            gesture VARCHAR(255),
                            FOREIGN KEY (location_id) REFERENCES Location(location_id)
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Alert (
                            alert_id INT AUTO_INCREMENT PRIMARY KEY,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            alert_type ENUM('Lone Woman', 'Surrounded by Men', 'SOS Gesture', 'Unusual Pattern'),
                            location_id INT,
                            details TEXT,
                            FOREIGN KEY (location_id) REFERENCES Location(location_id)
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Hotspot (
                            hotspot_id INT AUTO_INCREMENT PRIMARY KEY,
                            location_id INT,
                            crime_type ENUM('Rape', 'Kidnapping', 'Stalking', 'Other'),
                            incident_count INT,
                            hotspot_status ENUM('Yes', 'No'),
                            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (location_id) REFERENCES Location(location_id)
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Gender_Distribution (
                            distribution_id INT AUTO_INCREMENT PRIMARY KEY,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            location_id INT,
                            male_count INT,
                            female_count INT,
                            FOREIGN KEY (location_id) REFERENCES Location(location_id)
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS Crime_Incident (
                            incident_id INT AUTO_INCREMENT PRIMARY KEY,
                            location_id INT,
                            crime_type ENUM('Rape', 'Kidnapping', 'Stalking', 'Other'),
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (location_id) REFERENCES Location(location_id)
                        )''')
        
        db.commit()
        print("Tables created successfully.")
    except Error as e:
        print(f"Error: {e}")

def insert_data(db, table, data):
    try:
        cursor = db.cursor()
        
        # Remove AUTO_INCREMENT fields from data
        if table == 'Person' and 'person_id' in data:
            del data['person_id']
        elif table == 'Location' and 'location_id' in data:
            del data['location_id']
        elif table == 'Alert' and 'alert_id' in data:
            del data['alert_id']
        elif table == 'Hotspot' and 'hotspot_id' in data:
            del data['hotspot_id']
        elif table == 'Gender_Distribution' and 'distribution_id' in data:
            del data['distribution_id']
        elif table == 'Crime_Incident' and 'incident_id' in data:
            del data['incident_id']
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
        cursor.execute(sql, tuple(data.values()))
        db.commit()
        print(f"Data inserted into {table} successfully.")
    except Error as e:
        print(f"Error: {e}")

def view_data(db, table):
    try:
        cursor = db.cursor()
        cursor.execute(f'SELECT * FROM {table}')
        rows = cursor.fetchall()
        for row in rows:
            print(row)
    except Error as e:
        print(f"Error: {e}")

def update_data(db, table, updates, condition):
    try:
        cursor = db.cursor()
        set_clause = ', '.join([f"{col} = %s" for col in updates.keys()])
        sql = f'UPDATE {table} SET {set_clause} WHERE {condition}'
        cursor.execute(sql, tuple(updates.values()))
        db.commit()
        print(f"Data updated in {table} successfully.")
    except Error as e:
        print(f"Error: {e}")

def delete_data(db, table, condition):
    try:
        cursor = db.cursor()
        cursor.execute(f'DELETE FROM {table} WHERE {condition}')
        db.commit()
        print(f"Data deleted from {table} successfully.")
    except Error as e:
        print(f"Error: {e}")

def get_column_names(db, table):
    cursor = db.cursor()
    cursor.execute(f'SHOW COLUMNS FROM {table}')
    columns = [row[0] for row in cursor.fetchall()]
    return columns

def menu(db):
    while True:
        print("\nMenu:")
        print("1. View data from a table")
        print("2. Insert data into a table")
        print("3. Update data in a table")
        print("4. Delete data from a table")
        print("5. Exit")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            table = input("Enter the table name to view: ").strip()
            view_data(db, table)
        elif choice == '2':
            table = input("Enter the table name to insert data into: ").strip()
            columns = get_column_names(db, table)
            data = {}
            for column in columns:
                if column.endswith('_id'):  # Skip AUTO_INCREMENT columns
                    continue
                value = input(f"Enter value for {column} (type 'NULL' for null values): ").strip()
                if value.upper() == 'NULL':
                    data[column] = None
                else:
                    data[column] = value
            insert_data(db, table, data)
        elif choice == '3':
            table = input("Enter the table name to update: ").strip()
            updates = {}
            condition = input("Enter the condition for update (e.g., 'person_id = 1'): ").strip()
            while True:
                column = input("Enter column name to update (or type 'done' to finish): ").strip()
                if column.lower() == 'done':
                    break
                value = input(f"Enter new value for {column}: ").strip()
                updates[column] = value
            update_data(db, table, updates, condition)
        elif choice == '4':
            table = input("Enter the table name to delete from: ").strip()
            condition = input("Enter the condition for delete (e.g., 'person_id = 1'): ").strip()
            delete_data(db, table, condition)
        elif choice == '5':
            print("Exiting...")
            db.close()
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

def main():
    db = create_connection()
    
    if db:
        create_tables(db)
        menu(db)

if __name__ == '__main__':
    main()


"""
INT -> person_id, location_id, alert_id, hotspot_id, incident_count, distribution_id, male_count, female_count, incident_id
DATETIME -> timestamp, last_updated
ENUM -> gender, alert_type, crime_type, hotspot_status
VARCHAR(255) -> gesture, name, details
FLOAT -> latitude, longitude 
"""