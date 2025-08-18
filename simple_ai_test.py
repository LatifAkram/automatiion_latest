#!/usr/bin/env python3
import requests
import json

print("Testing AI endpoints...")

# Test Gemini
print("1. Testing Gemini...")
try:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyBb-AFGtxM2biSnESY85nyk-fdR74O153c"
    payload = {"contents":[{"parts":[{"text":"What is 2+2?"}]}]}
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Gemini WORKING")
    else:
        print(f"❌ Error: {response.text[:200]}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test Local LLM  
print("\n2. Testing Local LLM...")
try:
    payload = {
        "model": "qwen2-vl-7b-instruct",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False
    }
    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Local LLM WORKING")
    else:
        print(f"❌ Error: {response.text[:200]}")
except Exception as e:
    print(f"❌ Failed: {e}")