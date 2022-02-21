from collections import Counter
import os

p = os.listdir('słowniki/')
arr = []
for i in p:
    p2 = os.listdir("słowniki/"+i)
    for j in p2:
        with open('słowniki/'+i+"/"+j, 'r', encoding="utf-8") as f:
            file = f.read()
            for k in file.split(','):
                arr.append(k)

arr2 = []
for i in Counter(arr).most_common():
    if i[1] >= 50:
        arr2.append(i[0])

with open('restricted.txt', 'a', encoding = "utf-8" ) as f:
    f.write(",".join(arr2))

print(arr2)