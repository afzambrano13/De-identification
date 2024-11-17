# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:16:31 2024

@author: azamb
"""

import requests
import json

url = "http://localhost:11434/api/chat"

def llama3(prompt):
    data = {
        "model": "llama3.2",
        "messages": [
            {
              "role": "user",
              "content": prompt
            }
        ],
        "stream": False
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=data)
    return(response.json()['message']['content'])

response = llama3("Please edit the provided text by removing any personally identifiable information (PII). This includes names, company names, places of origin, current living locations, addresses, and social media links. Replace all removed PII with '[REDACTED]'. Ensure that the rest of the text remains unchanged, word for word. Maintain the original punctuation, quotation marks, spaces, and line breaks. If the text does not contain any PII, return it as is. Please do this process with the following post:\nHello everyone Nice websites, this is mine http://apledsd.wordpress.com")
print(response)