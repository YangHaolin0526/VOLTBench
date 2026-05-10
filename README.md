# VOLTBench

This repository contains the VOLTBench prompt instructions and generation scripts.

## Contents

- `Longen_instructions/`: English and Chinese prompt templates.
- `py_files/prompt_dict.py`: builds pickle prompt dictionaries from `Longen_instructions`.
- `py_files/*_Gen*.py`: local Hugging Face generation scripts.
- `py_files/LLM_Gen.py`: OpenAI-compatible API generation script using `OPENAI_API_KEY`.

## Examples

Build a prompt dictionary:

```bash
python py_files/prompt_dict.py --language EN --num-section 5 --word-section 200
python py_files/prompt_dict.py --language CH --num-section 5 --word-section 200
```

Run API generation:

```bash
export OPENAI_API_KEY="your_api_key"
python py_files/LLM_Gen.py --prompt-folder Longen_instructions/EN --model gpt-4o-mini
```

Run local Hugging Face generation:

```bash
python py_files/Qwen_7B_Gen_EN.py
python py_files/Qwen_7B_Gen.py
```
