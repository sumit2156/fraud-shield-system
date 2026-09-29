import sqlite3

def setup_database():
    conn = sqlite3.connect("banking_audit.db")
    cursor = conn.cursor()
    
    # Customer baseline profile table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        customer_name TEXT,
        home_city TEXT,
        avg_txn_amount REAL
    )""")
    
    # Access audit logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        action TEXT,
        ip_location TEXT,
        status TEXT
    )""")
    
    # Insert dummy user data
    cursor.execute("INSERT OR REPLACE INTO user_profiles VALUES ('USR_101', 'Rahul Sharma', 'Mumbai', 45.00)")
    
    # Insert suspicious logs for this user
    cursor.execute("INSERT OR REPLACE INTO audit_logs (user_id, action, ip_location, status) VALUES ('USR_101', 'login_attempt', 'Amsterdam, Netherlands', 'FAILED')")
    cursor.execute("INSERT OR REPLACE INTO audit_logs (user_id, action, ip_location, status) VALUES ('USR_101', 'login_attempt', 'Amsterdam, Netherlands', 'FAILED')")
    cursor.execute("INSERT OR REPLACE INTO audit_logs (user_id, action, ip_location, status) VALUES ('USR_101', 'password_reset', 'Amsterdam, Netherlands', 'SUCCESS')")
    
    conn.commit()
    conn.close()

# The tool function that the AI Agent will call
def query_user_security_logs(user_id: str):
    conn = sqlite3.connect("banking_audit.db")
    cursor = conn.cursor()
    cursor.execute("SELECT action, ip_location, status FROM audit_logs WHERE user_id = ?", (user_id,))
    logs = cursor.fetchall()
    conn.close()
    return logs

if __name__ == "__main__":
    setup_database()
    print("Database 'banking_audit.db' created and populated with forensic records.")