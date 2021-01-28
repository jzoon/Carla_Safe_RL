while True:
    try:
        exec(open("main.py").read())
    except Exception:
        continue
