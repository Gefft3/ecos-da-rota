import requests

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

data = '{\n  "model": "llama3.1",\n  "prompt":"Why is the sky blue?"\n}'

response = requests.post('http://10.88.8.31:0800/api/generate', headers=headers, data=data)