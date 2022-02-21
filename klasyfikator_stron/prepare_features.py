import os
import pickle

objects = []
with open("models/headers.pickle", "rb") as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

p = os.listdir("kategorie_cleaned/")
for i in p:
    print(i)
    p2 = os.listdir("kategorie_cleaned/"+i)
    for j in p2:
    	kp = os.listdir("k_dl/"+i)
    	if j in kp:
    		pass
    	else:
	        with open("kategorie_cleaned/"+i+"/"+j, 'r', encoding = "utf-8") as f:
	            file = f.read()
	        for k in file.split():
	            if k in objects[0][1:]:
	                if not os.path.exists("k_dl/" + i):
	                    os.makedirs("k_dl/" + i)
	                with open("k_dl/" + i + "/" + j, 'a', encoding = "utf-8") as f2:
	                    f2.write(k+" ")
	            else:
	                pass