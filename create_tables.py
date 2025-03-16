import pymysql
from sqlalchemy import create_engine, text

# Database connection details
DB_USER = "dbwriter2"  # Replace with your database username
DB_PASSWORD = "dikshit2025"  # Replace with your database password
DB_NAME = "chat-factory-db"  # Replace with your database name
PUBLIC_IP = "35.246.79.35"  # Public IP from the image
PORT = 3306  # Default MySQL port

# Create connection string
connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{PUBLIC_IP}:{PORT}/{DB_NAME}"

# Create engine
engine = create_engine(connection_url)

# # SQL to create a new table
# create_table_sql = """
# CREATE TABLE IF NOT EXISTS users (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     name VARCHAR(100),
#     email VARCHAR(100) UNIQUE,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
# """

create_production_log_sql = """
CREATE TABLE production_log (
    shift_id SERIAL PRIMARY KEY,
    date DATE,
    shift_name TEXT,
    units_produced INT
);
"""

create_machine_status_sql = """
CREATE TABLE machine_status (
    machine_id SERIAL PRIMARY KEY,
    machine_name TEXT,
    last_downtime DATE,
    downtime_duration INT,
    issue_description TEXT
);
"""

create_maintenance_log_sql = """
CREATE TABLE maintenance_logs (
    machine_id INT,
    last_maintenance DATE,
    maintenance_notes TEXT
);
"""

sql_commands = [create_production_log_sql, create_machine_status_sql, create_maintenance_log_sql]

# Execute SQL query

for sql_command in sql_commands:
    try:
        with engine.connect() as connection:
            connection.execute(text(sql_command))
            print("Table created successfully.")
    except Exception as e:
        print(f"Error: {e}")