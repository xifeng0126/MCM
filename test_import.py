import time
import requests

print("start!")
time.sleep(2)
print("end!")

response = requests.get("http://www.baidu.com")
print(response.content.decode("utf-8"))
