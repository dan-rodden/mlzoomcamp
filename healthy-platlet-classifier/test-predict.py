#!/usr/bin/env python
# coding: utf-8

import requests
import json

# open up the json sample and name the sample
sample_id='gsmXYZ123'

with open('data/rna_sample_1.json', 'r') as f_in:
    sample = json.load(f_in)


host="localhost:9696"
url=f'http://{host}/predict'

response = requests.post(url, json=sample).json()
print(response)

if response['is_healthy'] == 1:
    print(f"Sample {sample_id} considered healthy. Probability of healthy platlet: {response['probability']}")
else:
    print(f"Sample {sample_id} considered cancerous. Probability of cancerous platlet: {float(1 - response['probability'])}")
