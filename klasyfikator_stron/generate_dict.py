import os
from collections import Counter
from stop_words import get_stop_words

def generate_dict():
	p = "kategorie_cleaned/"
	path = os.listdir("kategorie_cleaned/")

	# p = "C:/Users/agwozdziej/Documents/Work_ML/url-classifier/deep-learning/DL_pl/Data_cleaned/"
	# path = os.listdir("C:/Users/agwozdziej/Documents/Work_ML/url-classifier/DL_pl/deep-learning/Data_cleaned/")
	with open('restricted2.txt', 'r', encoding="utf-8") as f:
		skip_file = f.read().split(',')

	counter = 0
	for i in path:
		arr = []
		arr2 = []
		path2 = os.listdir(p+i)
		for j in path2:
			with open(p+i+"/"+j, 'r', encoding = "utf-8") as f:
				txt = f.read()
			counter += 1
			txt = list(filter(lambda x: x not in get_stop_words('polish'), txt.split()))
			for j in txt:
				if len(j) <= 2 or len(j) >= 20:
					pass
				else:
					arr.append(j.lower())
		res = Counter(arr)
		x = res.most_common()
		# print(x[:20])
		counter = 0
		for k in x:
			if counter > 210:
				break
			else:
				if len(k[0]) <= 2 or len(k[0]) >= 20:
					pass
				else:
					if k[0] in skip_file:
						pass
					else:
						counter += 1
						arr2.append(k[0])

		result = ",".join(arr2)
		if not os.path.exists("słowniki/" + i):
			os.makedirs("słowniki/" + i)
		with open("słowniki/"+i+"/"+i+".txt", 'w', encoding="utf-8") as f:
			f.write(result)