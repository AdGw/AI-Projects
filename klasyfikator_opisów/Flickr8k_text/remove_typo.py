import os
from collections import Counter
with open("token_9000.txt", 'r', encoding="utf-8") as f:
    r = f.readlines()

res = ''
for i in r:
    res += i.split("\t")[1].replace("."," ").replace("\n", '').lower()

print(sorted(set(res.split())))
# print(Counter(res.split()))