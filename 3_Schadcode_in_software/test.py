from datetime import datetime

print(datetime.now())


time = datetime.now().strftime("%H_%M_%S")
print("time:", type(time))