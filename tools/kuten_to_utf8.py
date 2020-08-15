#!/usr/bin/env python


#\u3066 - te

class UnicodeDictionary():
	
	def __init__(self):
		self.dict = {}
		f = open('data/jis0208.txt', 'r')
		line = f.readline()
		while line:
			jis = line[2:2+4]
			unicd = line[16:16+4]
			unicd_ = 0
			try:
				ku = int(jis[0:2], 16) - 32
				unicd_ = chr(int(unicd,16))
				ten = int(jis[2:4], 16) - 32
			except:
				ku = 0
				ten = 0
				unicd_=0

			print("ku:", ku, ", ten:", ten, "\t JIS:", jis, ", Unicode:", unicd)
			line = f.readline()
		f.close()





if __name__ == '__main__':
	uni = UnicodeDictionary()
	#asd = uni.dict[(42,2)]


