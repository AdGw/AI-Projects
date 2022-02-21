from collections import Counter
from textblob import TextBlob

p1 = open("Flickr8k_text/original.txt", 'r', encoding = "utf-8")
f1 = p1.read()
c = 0
for i in f1.split("\n"):
	c+=1
	print(c)
	point = i.split("\t")
	translator = TextBlob(point[1])
	res_t = translator.translate(to="pl")
	w1 = open("plik.txt", 'a', encoding = "utf-8" )
	w1.write(point[0]+"\t"+str(res_t)+"\n")

