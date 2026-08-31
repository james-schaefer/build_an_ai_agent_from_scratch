# Build an AI From Scratch
# Modified Listing 2.7, pg 28-29
# Structured output with Pydantic
#
# from the root of the project:
# ```
#   uv sync
#   uv run python 02/listing_2.7.py
# ```

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pydantic import BaseModel
from litellm import completion

class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

response = completion(
        model="gpt-5-mini", 
        messages=[{"role":"user",
                   "content":"My name is John Smith.  My email address is: jsmith@ghost.com. stop asking, but my phone number is 505-555-1234"}], 
        response_format=ExtractedInfo
    )

result = response.choices[0].message.content

print("results: ")
print(result)
