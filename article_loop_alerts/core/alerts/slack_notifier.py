import requests
import json
from dotenv import load_dotenv

load_dotenv()

class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        if not self.webhook_url:
            raise ValueError("SLACK_WEBHOOK_URL environment variable is missing")

    def send(self, message: str):
        payload = {"text": message}
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Slack notification failed: {e}")

    