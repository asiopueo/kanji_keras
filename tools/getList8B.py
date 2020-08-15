
import sys
from database.database import Database


SIZE_X = 64
SIZE_Y = 63

SAMPLE_WIDTH = 512
TOTAL_RECORDS = 51200
TOTAL_CHARACTERS = 320
#TOTAL_RECORDS = 50560
#TOTAL_CHARACTERS = 316
NUMBER_SAMPLE = 160

DATA_FILE = "etlcdb/ETL8B/ETL8B2C1"
DB_FILE = "database/jisx0208.db"





with open(DATA_FILE, 'rb') as file_handle:
	byte_buffer = file_handle.read( (TOTAL_RECORDS+1)*SAMPLE_WIDTH )
	db = Database(DB_FILE)

	for counter in range(1, TOTAL_RECORDS, NUMBER_SAMPLE):
		data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
		JIS_code = data[2:4].hex()
		character = db.getCharacter(JIS_code)
		#character = db.getUnicode(JIS_code)
		print('No.', int((counter-1)/NUMBER_SAMPLE)+1, ':', character, JIS_code)



