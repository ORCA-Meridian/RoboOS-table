import requests

MASTER_URL = "http://127.0.0.1:5000/publish_task"

task = (
    "Clear the table: "
    "step 1 pick up the trash bag from the floor and place it on the table, "
    "step 2 bag all large items on the table into the trash bag (leave the white lobster box on table), "
    "step 3 fetch the towel via replay trajectory and place it on the table, "
    "step 4 use the towel to sweep lobster debris into the white box then bag the box."
)

response = requests.post(MASTER_URL, json={"task": task, "refresh": True})
print("Status :", response.status_code)
print("Response:", response.json())
