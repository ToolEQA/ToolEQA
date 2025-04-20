import json
import http
import base64
import http.client
import json
import PIL
import logging
from io import BytesIO
import time

class RequestAPI:
    def __init__(self):
        self.base_url = '<api url>'
        self.authorization = '<your token>'

    def convert_file_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def convert_PIL_to_base64(self, image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            base64_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return base64_image

    def convert_base64(self, image):
        if isinstance(image, str):
            base64_image = self.convert_file_to_base64(image)
        elif isinstance(image, PIL.Image.Image):
            base64_image = self.convert_PIL_to_base64(image)
        return base64_image

    def prepare_data(self, image, prompt, kb):        
        base64_image = self.convert_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        for item in kb:
            base64_image = self.convert_base64(item['image'])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": item['text']
                    }
                ]
            })

        return messages

    def request_with_retry(self, image, prompt, kb, retries=10):
        def exponential_backoff(attempt):
            return min(2 ** attempt, 60)

        for attempt in range(retries):
            try:
                return self.requests_api(image, prompt, kb)
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = exponential_backoff(attempt)
                    logging.log(logging.ERROR, f"Request failed, retrying in {wait_time} seconds... {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e

    def requests_api(self, image, prompt, kb):
        conn = http.client.HTTPSConnection(self.base_url)
        messages = self.prepare_data(image, prompt, kb)
        payload = json.dumps({
            "model": "gpt-4o",
            "stream": False,
            "messages": messages,
                "max_tokens": 400
            })
        headers = {
            'Authorization': self.authorization,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))

        return [data["choices"][0]["message"]["content"]]
    