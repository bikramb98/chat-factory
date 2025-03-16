import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Database connection details
DB_USER = "dbwriter2"
DB_PASSWORD = "dikshit2025"
DB_NAME = "chat-factory-db"
PUBLIC_IP = "35.246.79.35"
PORT = 3306

# Create connection string
connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{PUBLIC_IP}:{PORT}/{DB_NAME}"
engine = create_engine(connection_url)

def populate_machine_status():
    machines = ["CNC Lathe-1000", "Hydraulic Press-200", "Laser Cutter-X5"]
    issues = [
        "Coolant leakage", "Overheating motor", "Tool misalignment", "Excessive vibration"
    ]
    
    create_table_query = text("""
        CREATE TABLE IF NOT EXISTS machine_status_2 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            machine_name VARCHAR(255) NOT NULL,
            last_downtime DATETIME NOT NULL,
            downtime_duration INT NOT NULL,
            issue_description TEXT NOT NULL
        )
    """)
    
    with engine.connect() as connection:
        connection.execute(create_table_query)
        
        # Generate serial last_downtime values with some randomness in hours and minutes
        base_time = datetime.now() - timedelta(days=14)
        last_downtime_list = [
            base_time + timedelta(days=i, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            for i in range(20)
        ]
        last_downtime_list.sort()  # Ensure they remain in ascending order
        
        for i in range(20):
            machine_name = random.choice(machines)
            last_downtime = last_downtime_list[i]
            downtime_duration = random.randint(15, 180)  # Random downtime duration in minutes
            issue_description = random.choice(issues)
            
            insert_machine_status = text("""
                INSERT INTO machine_status_2 (machine_name, last_downtime, downtime_duration, issue_description)
                VALUES (:machine_name, :last_downtime, :downtime_duration, :issue_description)
            """)
            
            connection.execute(insert_machine_status, {
                "machine_name": machine_name,
                "last_downtime": last_downtime,
                "downtime_duration": downtime_duration,
                "issue_description": issue_description
            })
        
        connection.commit()
    
    print("Inserted 20 rows into machine_status_2 table.")

def populate_production_log():
    create_production_log_table = text("""
        CREATE TABLE IF NOT EXISTS production_log (
            data_id INT AUTO_INCREMENT PRIMARY KEY,
            production_date DATE NOT NULL,
            shift ENUM('morning_shift', 'afternoon_shift', 'night_shift') NOT NULL,
            units_produced INT NOT NULL
        )
    """)
    
    with engine.connect() as connection:
        connection.execute(create_production_log_table)
        
        # Get machine downtime data
        machine_downtime_query = text("SELECT last_downtime, downtime_duration FROM machine_status_2")
        result = connection.execute(machine_downtime_query)
        downtime_data = result.fetchall()
        
        base_date = datetime.now() - timedelta(days=14)
        shifts = {"morning_shift": (6, 14), "afternoon_shift": (14, 22), "night_shift": (22, 6)}
        
        for day in range(14):
            production_date = (base_date + timedelta(days=day)).date()
            for shift, (start_hour, end_hour) in shifts.items():
                shift_duration = 480  # 8 hours in minutes
                
                # Calculate downtime affecting this shift
                downtime_in_shift = sum(
                    min(downtime, shift_duration) 
                    for downtime_time, downtime in downtime_data
                    if downtime_time.date() == production_date and 
                    (start_hour <= downtime_time.hour < end_hour or (end_hour < start_hour and (downtime_time.hour >= start_hour or downtime_time.hour < end_hour)))
                )
                
                available_time = shift_duration - downtime_in_shift
                units_produced = max(0, (available_time // 2))  # Assuming 1 unit per 2 min when running
                
                insert_production_log = text("""
                    INSERT INTO production_log (production_date, shift, units_produced)
                    VALUES (:production_date, :shift, :units_produced)
                """)
                
                connection.execute(insert_production_log, {
                    "production_date": production_date,
                    "shift": shift,
                    "units_produced": units_produced
                })
        
        connection.commit()
    
    print("Inserted production data into production_log table.")

# ... existing code ...

def populate_inventory():
    parts = [
        "Bearing Assembly", "Hydraulic Seal", "Drive Belt", "Control Valve",
        "Pressure Gauge", "Circuit Board", "Motor Coupling", "Filter Element",
        "Gear Set", "Pneumatic Cylinder"
    ]
    shelves = ["Shelf-A", "Shelf-B", "Container-1", "Container-2"]
    
    create_inventory_table = text("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            part_name VARCHAR(255) NOT NULL,
            storage_location VARCHAR(50) NOT NULL,
            quantity INT NOT NULL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    with engine.connect() as connection:
        connection.execute(create_inventory_table)
        
        # Distribute parts across shelves with random quantities
        for part in parts:
            location = random.choice(shelves)
            quantity = random.randint(10, 100)
            
            insert_inventory = text("""
                INSERT INTO inventory (part_name, storage_location, quantity)
                VALUES (:part_name, :storage_location, :quantity)
            """)
            
            connection.execute(insert_inventory, {
                "part_name": part,
                "storage_location": location,
                "quantity": quantity
            })
        
        connection.commit()
    
    print("Inserted 10 parts into inventory table.")

# Update the final section to include the new function
# populate_machine_status()
# populate_production_log()
populate_inventory()

# # Run the functions
# populate_machine_status()
# populate_production_log()