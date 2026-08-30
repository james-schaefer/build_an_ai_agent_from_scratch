# Build an AI From Scratch
# Modified Listing 2.6, pg 27-28
# This listing demonstrates managing comversation history
#
# from the root of the project:
# ```
#   uv sync
#   uv run python 02/listing_2.6.py
# ```

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from litellm import completion

messages = []

print()

## First Exchange
messages.append({"role": "user", "content": "my name is james"})

response1 = completion( model="gpt-5-mini", messages=messages)

assistant_message1 = response1.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message1})
print(assistant_message1)
print()


## Second Exchange
messages.append({"role": "user", "content": "kindly say my name."})

response2 = completion( model="gpt-5-mini", messages=messages)

assistant_message2 = response2.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message2})
print(assistant_message2)
print()
