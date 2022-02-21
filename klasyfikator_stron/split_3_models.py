import os
import math

def split_models():
    p = os.listdir("kategorie_cleaned/")
    threshold = 4
    cc = 1
    for i in p:
        p2 = os.listdir("kategorie_cleaned/" + i)
        c = 0
        for j in p2:
            with open("kategorie_cleaned/" + i + "/" + j, 'r', encoding="utf-8") as f:
                file = f.read()
            if c >= int(len(p2)/3):
                cc += 1
                c = 0
            if cc >= threshold:
                pass
            else:
                if not os.path.exists("3_models/" + str(cc) + "/" + i):
                    os.makedirs("3_models/" + str(cc) + "/" + i)
                # print(cc)
                with open("3_models/" + str(cc) + "/" + i + "/" + j, 'w', encoding="utf-8") as f:
                    f.write(file)
                c += 1
        cc = 1
