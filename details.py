import sqlite3

con = sqlite3.connect('CANDIDATE_RESUME.db')
cursor = con.cursor()

cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

for row in rows:
    name, email, phone, score = row

con.commit()
con.close() 