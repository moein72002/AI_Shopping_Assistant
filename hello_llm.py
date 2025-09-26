import os
import time  # Import the time module
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
    # Record the start time
    start_time = time.time()

    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": """
                Hello! How are you? Who are you?
                """,
            }
        ],
        model="gpt-4o-mini",
    )

    # Record the end time
    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time

    # Print the response and the time taken
    print("LLM call successful!")
    print(chat_completion.choices[0].message.content)
    print(f"\nLLM response time: {duration:.2f} seconds") # Print the formatted duration

except Exception as e:
    print(f"An error occurred: {e}")