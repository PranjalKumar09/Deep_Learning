import psycopg2
from psycopg2 import Error

def create_connection():
    try:
        db = psycopg2.connect(
            host="localhost",
            database="sih",
            user="postgres",
            password="0907"
        )
        if db is not None:
            print("Successfully connected to PostgreSQL database")
        return db
    except Error as e:
        print(f"Error: {e}")
        return None

def create_tables(db):
    try:
        cursor = db.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Location (
                location_id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                latitude FLOAT,
                longitude FLOAT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Person (
                person_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                gender VARCHAR(10) CHECK (gender IN ('Male', 'Female')),
                location_id INT,
                gesture VARCHAR(255),
                FOREIGN KEY (location_id) REFERENCES Location(location_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Alert (
                alert_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_type VARCHAR(20) CHECK (alert_type IN ('Lone Woman', 'Surrounded by Men', 'SOS Gesture', 'Unusual Pattern')),
                location_id INT,
                details TEXT,
                FOREIGN KEY (location_id) REFERENCES Location(location_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Hotspot (
                hotspot_id SERIAL PRIMARY KEY,
                location_id INT,
                crime_type VARCHAR(20) CHECK (crime_type IN ('Rape', 'Kidnapping', 'Stalking', 'Other')),
                incident_count INT,
                hotspot_status VARCHAR(3) CHECK (hotspot_status IN ('Yes', 'No')),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES Location(location_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Gender_Distribution (
                distribution_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location_id INT,
                male_count INT,
                female_count INT,
                FOREIGN KEY (location_id) REFERENCES Location(location_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Crime_Incident (
                incident_id SERIAL PRIMARY KEY,
                location_id INT,
                crime_type VARCHAR(20) CHECK (crime_type IN ('Rape', 'Kidnapping', 'Stalking', 'Other')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES Location(location_id)
            )
        ''')
        
        db.commit()
        print("Tables created successfully.")
    except Error as e:
        print(f"Error: {e}")

def insert_data(db, table, data):
    try:
        cursor = db.cursor()
        
        # Remove SERIAL fields from data
        serial_columns = {
            'Person': 'person_id',
            'Location': 'location_id',
            'Alert': 'alert_id',
            'Hotspot': 'hotspot_id',
            'Gender_Distribution': 'distribution_id',
            'Crime_Incident': 'incident_id'
        }
        
        if table in serial_columns and serial_columns[table] in data:
            del data[serial_columns[table]]
        
        if not data:
            raise ValueError("No data to insert.")
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        sql = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
        
        print(f"Executing SQL: {sql}")
        print(f"Data: {tuple(data.values())}")
        
        cursor.execute(sql, tuple(data.values()))
        db.commit()
        print(f"Data inserted into {table} successfully.")
    except Error as e:
        print(f"Error: {e}")
    except ValueError as ve:
        print(f"Value Error: {ve}")

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
    try:
        cursor = db.cursor()
        query = '''
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %s
        '''
        cursor.execute(query, (table,))
        columns = [row[0] for row in cursor.fetchall()]
        return columns
    except Error as e:
        print(f"Error: {e}")
        return []

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
            if not columns:
                print(f"Table '{table}' does not exist or could not retrieve columns.")
                continue
            data = {}
            for column in columns:
                if column.endswith('_id'):  # Skip SERIAL columns
                    continue
                value = input(f"Enter value for {column} (type 'NULL' for null values): ").strip()
                if value.upper() == 'NULL':
                    data[column] = None
                else:
                    data[column] = value
            # Debugging: Print the collected data
            print(f"Collected data: {data}")
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
