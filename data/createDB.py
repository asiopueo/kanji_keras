import sqlite3




CREATE_TABLE = 	'''CREATE TABLE JISX0208 (
					SHIFTJIS CHAR(6) NOT NULL,
					JISX0208 CHAR(6) NOT NULL,
					UNICODE CHAR(6) NOT NULL,
					KU INT,
					TEN INT,
					NAME CHAR(50)
				);
				'''

INSERT_SQL =	'''INSERT INTO JISX0208(SHIFTJIS, JISX0208, UNICODE, KU, TEN, NAME) VALUES (?,?,?,?,?,?);
				'''




def create_connection(db_file):
	try:
		ctx = sqlite3.connect(db_file)
		return ctx
	except sqlite3.Error as e:
		print(e)
		exit(1)



def create_table(ctx):
	cursor = ctx.cursor()
	try:
		cursor.execute(CREATE_TABLE)
	except sqlite3.Error as e:
		print(e)


def insert_values(ctx, source):
	crs = ctx.cursor()
	with open(source, 'r') as f:
		for line in f:
			if line[0]!='#':
				line_ = line.split('\t')
				shiftjis = line_[0][2:]
				jisx0208 = line_[1][2:]
				unicd = line_[2][2:]
				name = line_[3][2:-1]

				ku = int(jisx0208[0:2], 16) - 32
				ten = int(jisx0208[2:4], 16) - 32

				print (shiftjis, jisx0208, unicd, ku, ten, name)
				try:
					crs.execute(INSERT_SQL, (shiftjis, jisx0208, unicd, ku, ten, name))
				except Error as e:
					print(e)

	ctx.commit() # 'zis was the culprit!'




def main():
	db_file = './jisx0208.db'
	jis_file = './jis0208.txt'
	ctx = create_connection(db_file)
	create_table(ctx)
	insert_values(ctx, jis_file)



if __name__=='__main__':
	main()