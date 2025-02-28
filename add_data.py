from datetime import datetime, timedelta
import pymysql
import random
from sqlalchemy import create_engine, text

# Database connection details
DB_USER = "dbwriter2"  # Your database username
DB_PASSWORD = "dikshit2025"  # Your database password
DB_NAME = "chat-factory-db"  # Your database name
PUBLIC_IP = "35.246.79.35"  # Public IP from your Cloud SQL instance
PORT = 3306  # Default MySQL port

# Create connection string
connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{PUBLIC_IP}:{PORT}/{DB_NAME}"

# Create engine
engine = create_engine(connection_url)

try:
    with engine.connect() as connection:
        # Start a transaction
        transaction = connection.begin()
        
        # Insert into `production_log`
        insert_production_log = text("""
            INSERT INTO production_log (date, shift_name, units_produced)
            VALUES (:date, :shift_name, :units_produced)
        """)
        connection.execute(insert_production_log, {
            "date": datetime.now().date(),
            "shift_name": "Morning Shift",
            "units_produced": random.randint(3000, 5000)
        })

        # Insert into `machine_status`
        insert_machine_status = text("""
            INSERT INTO machine_status (machine_name, last_downtime, downtime_duration, issue_description)
            VALUES (:machine_name, :last_downtime, :downtime_duration, :issue_description)
        """)
        connection.execute(insert_machine_status, {
            "machine_name": "Conveyor A7",
            "last_downtime": (datetime.now() - timedelta(days=1)).date(),
            "downtime_duration": 30,
            "issue_description": "Belt misalignment"
        })

        # Commit the transaction
        transaction.commit()
        print("Data inserted successfully.")

except Exception as e:
    print(f"Error: {e}")

finally:
    engine.dispose()  # Close connection pool