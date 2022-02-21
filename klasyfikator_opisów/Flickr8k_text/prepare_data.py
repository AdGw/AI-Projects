import os
with open("train_9000.txt", 'r', encoding="utf-8") as f:
    r1 = f.readlines()
with open("test_9000.txt", 'r', encoding="utf-8") as f2:
    r2 = f2.readlines()
arr = []
for i in r1:
    arr.append(i.replace("\n", ""))
for i in r2:
    arr.append(i.replace("\n", ""))

p = os.listdir("data_all/")
c = 0
for i in p:
    if i in arr:
        os.remove("data_all/"+i)
    else:
        c += 1
        print(c)
# print(len(r2))
# s = r1+r2
# print(s)
# arr = []
# for i in r:
#     arr.append(i.split("\t")[0][:-2])
# print(len(arr))

# arr = set(arr)

# for i in arr:
#     with open('check.txt', 'a', encoding="utf-8") as w:
#         w.write(i+"\n")