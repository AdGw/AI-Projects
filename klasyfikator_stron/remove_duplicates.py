import os

def remove_duplicates():
    path = "kategorie_cleaned/"
    listed = os.listdir(path)
    for i in listed:
        arr = []
        listed_txt = os.listdir(path + str(i))
        for j in listed_txt:
            with open(path + str(i) + "/" + j, "r", encoding="utf-8-sig") as file:
                x = file.read()
            if x in arr:
                os.remove(path + str(i) + "/" + j)
            else:
                arr.append(x)
