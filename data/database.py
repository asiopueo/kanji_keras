import sqlite3
from sqlite3 import Error


def create_connection(db_file):
	try:
		ctx = sqlite3.connect(db_file)
		return ctx
	except Error as e:
		print(e)

	return None


def getUnicode(ctx, index):
	cursor = ctx.cursor()
	cursor.execute("SELECT UNICODE FROM JISX0208 WHERE JISX0208 LIKE ?;", (index,))
	return str(*cursor.fetchone())





if __name__=='__main__':
	database = './testDB.db'
	ctx = create_connection(database)

	with ctx:
		select_task_by_priority(ctx, 1)
		select_all_tasks(ctx)
