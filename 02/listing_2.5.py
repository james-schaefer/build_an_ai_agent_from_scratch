# Build an AI From Scratch
# Modified Listing 2.5, pg 27
# This listing demonstrates the stateless nature of consecutive LLM calls.
#
# from the root of the project:
# ```
#   uv sync
#   uv run python 02/listing_2.5.py
# ```

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from litellm import completion

#First call


response1 = completion(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
)

response2 = completion(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "Hello, my name is james s."}
        ]
)

response3 = completion(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "Kindly say my name."}
        ]
)

print(response1.choices[0].message.content)
print()
print(response2.choices[0].message.content)
print()
print(response3.choices[0].message.content)
