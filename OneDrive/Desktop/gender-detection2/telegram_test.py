import requests

BOT_TOKEN = "8424334136:AAFyp3bBumrAOwYhERdL4gYxxLXK0QZI_NY"
CHAT_ID = "976435954"

message = """
🚨 WOMEN SAFETY ALERT 🚨

Potential emergency detected.

✔ Female detected
✔ Help gesture detected

Immediate attention required.
"""

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

data = {
    "chat_id": CHAT_ID,
    "text": message
}

response = requests.post(url, data=data)

print(response.json())