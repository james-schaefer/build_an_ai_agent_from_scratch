# Build an AI From Scratch
# Combined Listing 2.17 (pg 36-41), which gathers together listings 2.10,
# 2.11, 2.13, 2.14, 2.15, 2.16, and 2.17 from Chapter 2.
#
#   2.10  load GAIA Level 1 problems          -> load_gaia_level1()
#   2.11  GaiaOutput schema + system prompt   -> GaiaOutput, GAIA_PROMPT
#   2.13  exact-match answer checking          -> is_correct()
#   2.14  solve a single problem (LLM call)    -> solve_problem()
#   2.15  evaluate one problem/model pair       -> evaluate_gaia_single()
#   2.16  run the experiment across all pairs  -> run_experiment()
#   2.17  select models/subset + print results  -> main() (CLI driver)
#
# ```
#   uv sync
#   uv run python 02/listing_2.17.py --help
#   uv run python 02/listing_2.17.py --limit 20 --models gpt-5.5,gpt-5.4-mini,anthropic/claude-sonnet-4-6,anthropic/claude-haiku-4-5
# ```

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import argparse
import asyncio

from pydantic import BaseModel
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio


# ----------------------------------------------------------------------
# Listing 2.11 -- structured output schema + GAIA system prompt
# ----------------------------------------------------------------------

class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

GAIA_PROMPT = """You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."""


# ----------------------------------------------------------------------
# Supporting code -- single concurrency semaphore (a la Listing 2.9)
# ----------------------------------------------------------------------

# Limit concurrent LLM calls.  The book's Listing 2.12 splits this per
# provider (30 for OpenAI, 10 for Anthropic); that listing is out of scope
# here, so we use one shared semaphore instead.
semaphore = asyncio.Semaphore(10)


def get_provider(model: str) -> str:
    """Extract provider name from a LiteLLM model string."""
    return "anthropic" if model.startswith("anthropic/") else "openai"


# ----------------------------------------------------------------------
# Listing 2.13 -- exact-match answer checking
# ----------------------------------------------------------------------

def is_correct(prediction: str | None, answer: str) -> bool:
    """Case-insensitive exact match between prediction and ground truth."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


# ----------------------------------------------------------------------
# Listing 2.14 -- solve a single problem with a structured LLM call
# ----------------------------------------------------------------------

async def solve_problem(model: str, question: str) -> GaiaOutput:
    """Solve one GAIA problem and return the structured GaiaOutput."""
    async with semaphore:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": GAIA_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )

    finish_reason = response.choices[0].finish_reason
    content = response.choices[0].message.content

    if finish_reason == "refusal" or content is None:
        return GaiaOutput(
            is_solvable=False,
            unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
            final_answer="",
        )
    return GaiaOutput.model_validate_json(content)


# ----------------------------------------------------------------------
# Listing 2.15 -- evaluate a single problem/model pair
# ----------------------------------------------------------------------

async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    """Evaluate one problem for one model and return a result record."""
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }


# ----------------------------------------------------------------------
# Listing 2.16 -- run the experiment across all problem/model pairs
# ----------------------------------------------------------------------

async def run_experiment(
    problems: list[dict],
    models: list[str],
) -> dict[str, list]:
    """Evaluate every model on every problem and group results by model."""
    tasks = [
        evaluate_gaia_single(problem, model)
        for problem in problems
        for model in models
    ]

    all_results = await tqdm_asyncio.gather(*tasks)

    results = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)

    return results


# ----------------------------------------------------------------------
# Listing 2.10 -- load the GAIA Level 1 validation split
# ----------------------------------------------------------------------

def load_gaia_level1(data_dir: str | None = "./data/gaia"):
    """Load the GAIA 2023 Level 1 validation split.

    Prefers an on-disk copy at ``data_dir`` (the layout produced by
    ``hf download gaia-benchmark/GAIA --local-dir <data_dir>``); the configs
    are defined in ``<data_dir>/README.md`` so ``load_dataset`` resolves the
    ``2023_level1`` / ``validation`` split from the local parquet files.
    Falls back to the HuggingFace Hub repo id ``gaia-benchmark/GAIA`` when
    ``data_dir`` is not provided or does not exist on disk.

    Returns a HuggingFace Dataset.  Raises SystemExit with a helpful message
    if the `datasets` library or the dataset / HF_TOKEN is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - dependency is in pyproject
        raise SystemExit(
            "The `datasets` package is required.  Run `uv sync` first."
        ) from e

    import os
    source = data_dir if data_dir and os.path.isdir(data_dir) else "gaia-benchmark/GAIA"
    using_local = source != "gaia-benchmark/GAIA"

    try:
        return load_dataset(source, "2023_level1", split="validation")
    except Exception as e:
        where = f"local dir {source}" if using_local else "the HuggingFace Hub"
        raise SystemExit(
            f"Could not load the GAIA dataset from {where}.\n"
            "If loading from the Hub, this usually means the dataset is not available yet or HF_TOKEN\n"
            "is missing from your .env (see .env.example).\n"
            f"Underlying error: {e}"
        ) from e


# ----------------------------------------------------------------------
# Listing 2.17 -- select models/subset, run the experiment, print results
# ----------------------------------------------------------------------
#
# The notebook's 2.17 cell picks a fixed list of models and a 20-problem
# subset, then prints per-model accuracy.  Here it becomes an argparse-driven
# main() so the same experiment is runnable from the command line.  The
# default --models value reproduces the book's model list exactly, and the
# default --limit of 20 matches the book's subset size.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 2 GAIA benchmark evaluation.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-5.5,gpt-5.4-mini,anthropic/claude-sonnet-4-6,anthropic/claude-haiku-4-5",
        help="Comma-separated LiteLLM model names to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of Level 1 problems to evaluate (default: 20).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/gaia",
        help="Local GAIA repo dir from `hf download` (default: ./data/gaia). "
             "Falls back to the HuggingFace Hub if the dir is absent.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent LLM calls (default: 10).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models specified via --models.")

    # Size the shared semaphore to the requested concurrency.
    global semaphore
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"Loading GAIA Level 1 validation split ...")
    level1 = load_gaia_level1(data_dir=args.data_dir)
    print(f"Number of Level 1 problems available: {len(level1)}")

    limit = min(args.limit, len(level1))
    subset = list(level1.select(range(limit)))
    print(f"Evaluating {limit} problem(s) across {len(models)} model(s): {models}")
    print()

    results = await run_experiment(subset, models)

    # Per-model accuracy.
    for model in models:
        model_results = results[model]
        correct = sum(1 for r in model_results if r["correct"])
        solvable = sum(1 for r in model_results if r.get("is_solvable"))
        total = len(model_results)
        print(f"{model}: {correct}/{total} correct, {solvable}/{total} judged solvable")


## call main
if __name__ == "__main__":
    asyncio.run(main())
