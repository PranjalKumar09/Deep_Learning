import psycopg2
from psycopg2 import OperationalError

def create_connection():
    try:
        connection = psycopg2.connect(
            host="172.25.178.98",
            database="sih",
            user="postgres",
            password="0907",
            port="5432"
        )
        print("Connection successful")
        return connection
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None

def test_connection():
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT version();")
            db_version = cursor.fetchone()
            print("PostgreSQL version:", db_version)
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()
            connection.close()

test_connection()
