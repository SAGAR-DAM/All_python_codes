import sqlite3

# Connect to the database
conn = sqlite3.connect(r"D:\temporary backup\Codes\DBMS\sqlite3\my_database2.db")
cursor = conn.cursor()

# Create the table (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS users2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE
)
""")

# Sample dataset
users = [
    ("Sagar", "sagardam@example.com"),
    ("Alice", "alice@example.com"),
    ("Bob", "bob@example.com"),
    ("Charlie", "charlie@example.com"),
    ("David", "david@example.com"),
    ("Emma", "emma@example.com"),
    ("Frank", "frank@example.com"),
    ("Grace", "grace@example.com"),
    ("Hannah", "hannah@example.com"),
    ("Isaac", "isaac@example.com"),
    ("Jack", "jack@example.com")
]

# Insert only if the email does not already exist
for user in users:
    cursor.execute("SELECT * FROM users2 WHERE email = ?", (user[1],))
    if not cursor.fetchone():  # If no existing record, insert
        cursor.execute("INSERT INTO users2 (name, email) VALUES (?, ?)", user)
        
cursor.execute("INSERT INTO users2 (name, email) VALUES ('alu', 'alu@example.com')")
# cursor.execute("INSERT INTO users (name, email) VALUES (NULL, 'valu@example.com')")
# cursor.execute("INSERT INTO users (name, email) VALUES ('valu', NULL)")

    

# Commit changes
conn.commit()

# Fetch and print all stored users
cursor.execute("SELECT * FROM users")
cursor.execute("SELECT * FROM users ORDER BY id DESC")

rows = cursor.fetchall()

print("Stored Users:")
for row in rows:
    print(row)

# Close the connection
conn.close()
