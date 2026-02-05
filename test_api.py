"""
Test script to verify OpenAI API connection
"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Check if API key is loaded
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"✅ API Key loaded: {api_key[:20]}...{api_key[-10:]}")
else:
    print("❌ API Key NOT loaded")
    exit(1)

# Test OpenAI connection
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Make a simple test call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'API connection successful' in exactly 3 words."}
        ],
        max_tokens=10
    )
    
    result = response.choices[0].message.content
    print(f"✅ OpenAI API Response: {result}")
    print("✅ API connection is working!")
    
except Exception as e:
    print(f"❌ API Error: {str(e)}")
    exit(1)
