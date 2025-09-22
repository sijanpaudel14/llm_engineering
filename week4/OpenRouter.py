import requests
import json
from IPython.display import display, Markdown

class OpenRouter:
    MODEL = "qwen/qwen-2.5-coder-32b-instruct:free"
    def __init__(self, api_key: str, system_message="You are a helpful assistant."):
        self.api_key = api_key
        self.system_message = system_message
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _stream_generate(self, prompt: str):
        """Generator that yields tokens from a streaming response."""
        payload = {
            "model": self.MODEL,
            "messages": [{"role":"system", "content":self.system_message},
                         {"role": "user", "content": prompt}
                             ],
            "stream": True
        }

        buffer = ""
        with requests.post(self.base_url, headers=self.headers, json=payload, stream=True) as r:
            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                buffer += chunk
                while True:
                    line_end = buffer.find('\n')
                    if line_end == -1:
                        break
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        try:
                            data_obj = json.loads(data)
                            content = data_obj["choices"][0]["delta"].get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass

    def chat_with_stream_display(self, prompt: str):
        display_handle = display(Markdown(""), display_id=True)
        full_response = ""
    
        def generator():
            nonlocal full_response
            for chunk in self._stream_generate(prompt):
                full_response += chunk
                display_handle.update(Markdown(full_response))
                yield chunk
    
        return generator()
    def chat(self, prompt: str):
        """Generate the full response without streaming."""
        payload = {
            "model": self.MODEL,
            "messages": [{"role":"system", "content":self.system_message},
                         {"role": "user", "content": prompt}
                             ],
            "stream": False
        }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

