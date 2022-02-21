import csv
import os
import re

def generate():
    path = os.listdir("3_models/")
    for i in path:
        listed = os.listdir('3_models/'+i)
        with open('models/model_'+str(i)+'.csv', mode='w', newline='', encoding="utf-8-sig") as employee_file:
            employee_writer = csv.writer(employee_file)
            employee_writer.writerow(["Category", "Description"])
            for j in listed:
                # print(j)
                listed_txt = os.listdir("3_models/" + i + "/" + j)
                for k in listed_txt:
                    with open("3_models/" + str(i) + "/" + j + '/' + k, "r",encoding="utf-8-sig") as f:
                        text_original = f.read()
                    employee_writer.writerow([j, text_original])


