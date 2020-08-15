import sqlite3


sqlite_file = 'asd.sqlite'

conn = sqlite3.connect(sqlite_file)
c = conn.cursor()
conn.close()


