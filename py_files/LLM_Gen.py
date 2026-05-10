import argparse
import json
import os
import pickle
import re
from typing import Dict, Tuple

from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = "You are a helpful assistant skilled in generating long-form text."


def count_words(text: str) -> int:
    text_without_spaces = text.replace(" ", "")
    english_words = re.findall(r"\b[a-zA-Z]+\b", text_without_spaces)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text_without_spaces)
    other_chars = re.sub(r"\b[a-zA-Z]+\b|[\u4e00-\u9fff]", "", text_without_spaces)
    return len(english_words) + len(chinese_chars) + len(other_chars)


def call_model(client: OpenAI, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content or ""


def run_model_on_prompts(
    client: OpenAI,
    prompt_dict: Dict[str, Dict[str, str]],
    model_name: str,
    num_runs: int,
    output_prefix: str,
    prompt_folder: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_output_chars: int,
) -> Tuple[dict, dict]:
    word_count_filename = f"{prompt_folder}/{output_prefix}_word_count.json"
    output_filename = f"{prompt_folder}/{output_prefix}_output.json"

    if os.path.exists(output_filename) and os.path.exists(word_count_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            output_dict = json.load(f)
        with open(word_count_filename, "r", encoding="utf-8") as f:
            word_count_dict = json.load(f)
    else:
        output_dict = {}
        word_count_dict = {}

    for folder_key, files_dict in tqdm(prompt_dict.items(), desc="Processing prompts"):
        for file_name, prompt_content in files_dict.items():
            prompt_key = f"{folder_key}/{file_name}"
            output_dict.setdefault(prompt_key, [""] * num_runs)
            word_count_dict.setdefault(prompt_key, [0] * num_runs)

            if len(output_dict[prompt_key]) < num_runs:
                output_dict[prompt_key].extend([""] * (num_runs - len(output_dict[prompt_key])))
                word_count_dict[prompt_key].extend([0] * (num_runs - len(word_count_dict[prompt_key])))

            for run in range(num_runs):
                if len(output_dict[prompt_key][run]) >= min_output_chars:
                    print(f"Skipping run {run + 1} for {prompt_key} (already valid)")
                    continue

                success = False
                attempts = 0
                max_attempts = 5

                while not success and attempts < max_attempts:
                    try:
                        print(f"Running {prompt_key}, run {run + 1}...")
                        output = call_model(client, model_name, prompt_content, max_tokens, temperature, top_p)
                        if len(output) < min_output_chars:
                            raise ValueError("Output too short")

                        output_dict[prompt_key][run] = output
                        word_count = count_words(output)
                        word_count_dict[prompt_key][run] = word_count
                        success = True
                        print(f"Run {run + 1} successful. Model output {word_count} words.")

                        with open(word_count_filename, "w", encoding="utf-8") as f:
                            json.dump(word_count_dict, f, ensure_ascii=False, indent=4)
                        with open(output_filename, "w", encoding="utf-8") as f:
                            json.dump(output_dict, f, ensure_ascii=False, indent=4)
                    except Exception as e:
                        attempts += 1
                        print(f"Run {run + 1} failed (attempt {attempts}): {e}")
                        if attempts >= max_attempts:
                            print(f"Max attempts reached for {prompt_key}, run {run + 1}.")
                            output_dict[prompt_key][run] = ""
                            word_count_dict[prompt_key][run] = 0

    return output_dict, word_count_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VOLTBench outputs with an OpenAI-compatible API key.")
    parser.add_argument("--prompt-folder", default="Longen_instructions/EN", help="Folder containing *_dict.pkl files.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="API model name.")
    parser.add_argument("--model-shortname", default=None, help="Short name used in output files.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-output-chars", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    model_shortname = args.model_shortname or args.model.replace("/", "-")

    for pkl_file in os.listdir(args.prompt_folder):
        if not pkl_file.endswith(".pkl"):
            continue

        pkl_path = os.path.join(args.prompt_folder, pkl_file)
        with open(pkl_path, "rb") as f:
            loaded_prompt_dict = pickle.load(f)

        print(f"Processing {pkl_file}...")
        pkl_name = os.path.splitext(pkl_file)[0]
        output_prefix = f"{pkl_name}_{model_shortname}"
        run_model_on_prompts(
            client=client,
            prompt_dict=loaded_prompt_dict,
            model_name=args.model,
            num_runs=args.num_runs,
            output_prefix=output_prefix,
            prompt_folder=args.prompt_folder,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_output_chars=args.min_output_chars,
        )
        print(f"Finished processing {pkl_file}.")
