import time

duration = 100  # 10 minutes in seconds
interval = 10  # 100 seconds

end_time = time.time() + duration
while time.time() < end_time:
    print("hello")
    time.sleep(interval)
