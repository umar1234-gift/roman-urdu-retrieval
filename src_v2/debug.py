"""
Debug MQ0006 — Check finish_reason and raw response
"""
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

query = "What is the start date of Fall 2026 ka semester?"

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": f"""Translate the following query into natural English.

Rules:
1. Preserve the EXACT information need.
2. If it's already English, return it unchanged.
3. Return ONLY the translation, no explanation.

Query: {query}

Translation:"""}],
    temperature=0.0,
    max_tokens=1500,
)

print("finish_reason:", response.choices[0].finish_reason)
print("content:", repr(response.choices[0].message.content))
print("usage:", response.usage)