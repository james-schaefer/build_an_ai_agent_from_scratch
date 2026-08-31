# Build an AI From Scratch
# Modified Listing 2.9, pg 28-29
# Rate limit concurrent requests with a Semophare
#
# I had to make modifications to get this code to run, I also modified the
# output format.
#
# from the root of the project:
# ```
#   uv sync
#   uv run python 02/listing_2.9.py
# ```

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio 
from litellm import acompletion
from pydantic import BaseModel

# Limit to 10 concurrent requests
semaphore = asyncio.Semaphore(10)

async def call_llm(prompt: str) -> str: 
    """LLM with rate limiting and automatic retry."""
    async with semaphore:
        response = await acompletion(
            model="gpt-5-mini", 
            messages=[{"role":"user", "content": prompt}], 
            num_retries=3 # Automatic retry with exponential backoff
        )
    return response.choices[0].message.content

async def main():
   # Even with 100 concurrent tasks, only 10 API calls run at a time.
   prompts = [f"What is {i} + {i} =" for i in range(100)]

   tasks = [call_llm(p) for p in prompts]
   results = await asyncio.gather(*tasks, return_exceptions=True)

   for prompt, result in zip(prompts, results):
       print(f"User prompt: {prompt}")
       print(f"Response: {result}")
       print()

## call main
asyncio.run(main())
