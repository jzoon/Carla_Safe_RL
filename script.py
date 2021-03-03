i = 0

while i < 5:
    try:
        exec(open("main.py").read())
        i += 1
    except Exception as e:
        print(e)
        continue
