import json, urllib.request
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyA4qLjDDBz29iTlyFAzGWIHBl3oMZ48D5E"
data = json.dumps({"contents": [{"parts":[{"text": "hello"}]}]}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except Exception as e:
    print(e.read().decode() if hasattr(e, 'read') else str(e))
