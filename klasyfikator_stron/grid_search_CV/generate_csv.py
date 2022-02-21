import csv
import os
import re


path = os.listdir("Data/")
with open('model.csv', mode='w', newline='', encoding="utf-8-sig") as employee_file:
    employee_writer = csv.writer(employee_file)
    employee_writer.writerow(["Category", "Description"])
    for i in path:
        listed = os.listdir('Data/'+i)
        for j in listed:
            # print(j)
            with open("Data/" + str(i) + "/" + j, "r",encoding="utf-8-sig") as f:
                text_original = f.read()
            employee_writer.writerow([i, text_original])


