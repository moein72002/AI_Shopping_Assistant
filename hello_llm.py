import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key and Torob proxy URL from environment variables
api_key = os.getenv("OPENAI_API_KEY")
proxy_url = os.getenv("TOROB_PROXY_URL")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
if not proxy_url:
    raise ValueError("TOROB_PROXY_URL not found in .env file")

# Initialize the OpenAI client
# Note: The proxy URL should be passed to the httpx client
client = OpenAI(
    api_key=api_key,
    base_url=proxy_url,
)

try:
    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello! How are you? Who are you?",
            }
        ],
        model="gpt-5-nano",
    )

    # Print the response
    print("LLM call successful!")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
