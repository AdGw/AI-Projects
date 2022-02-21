import os
import threading, queue

with open("skip_fonts.txt", "r", encoding="utf-8-sig") as f:
    skip_fonts = f.read()


def char_replace(text):
    for c in text:
        if not c.isalpha():
            text = text.replace(c, " ")
        else:
            pass
    return text


arr = []


def main(i):
    counter = 0
    listed_txt = os.listdir("kategorie/" + str(i))
    for j in listed_txt:
        with open("kategorie/" + str(i) + "/" + j, "r", encoding="utf-8") as f:
            text_original = f.read()
        lines = (line for line in text_original.split() if
                 not line.count("https://") and not line.count("www.") and not line.count("https://"))
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        line = ' '.join(char_replace(chunk.lower()) for chunk in chunks if 3 <= len(chunk) < 20)
        line = line.translate(str.maketrans(
            ' ', ' ',
            skip_fonts)).lower()
        s = ''
        for l in line:
            if l in arr:
                pass
            else:
                arr.append(l)
                with open("cc.txt", 'a', encoding="utf-8") as f:
                    f.write(l)

        for k in line.split(" "):
            if len(k) > 20:
                pass
            elif len(k) <= 2:
                pass
            else:
                s += k + " "
        if counter < 2575:
            if not os.path.exists("kategorie_cleaned/" + i):
                os.makedirs("kategorie_cleaned/" + i)
            with open("kategorie_cleaned/" + i + "/" + "cleaned_original_" + j, "w", encoding="utf-8") as f:
                f.write(s)
            counter += 1
        else:
            break


q = queue.Queue()


def worker():
    while True:
        try:
            item = q.get()
            item_path = item["path"]
            main(item_path)
            q.task_done()
        except Exception:
            print("-pass err")
            pass


for i in range(1):
    threading.Thread(target=worker, daemon=True).start()


def format_data():
    cnt = 0
    c = 0

    path_doc = os.listdir("kategorie/")
    while cnt < len(path_doc):
        try:
            c += 1
            d = {
                "path": path_doc[cnt]
            }
            q.put(d)
            cnt += 1
        except queue.Full as e:
            print("-full")
            pass
    print('All task requests sent\n', end='')
    q.join()
    print('All work completed')
