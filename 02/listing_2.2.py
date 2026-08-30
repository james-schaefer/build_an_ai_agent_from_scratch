# Build an AI From Scratch
# Listing 2.2, pg 23
#
# from the root of the project:
# ```
#   uv sync
#   uv run python 02/listing_2.2.py
# ```

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpfull assistant."},
            {"role": "user", "content": "What is the capitol of France?"}
        ]
)

print(response.choices[0].message.content)
