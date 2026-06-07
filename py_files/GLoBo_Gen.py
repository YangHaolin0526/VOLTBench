import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


SYSTEM_PROMPT = "You are a helpful assistant skilled in generating long-form text."


def count_words(text: str) -> int:
    text_without_spaces = text.replace(" ", "")
    english_words = re.findall(r"\b[a-zA-Z]+\b", text_without_spaces)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text_without_spaces)
    other_chars = re.sub(r"\b[a-zA-Z]+\b|[\u4e00-\u9fff]", "", text_without_spaces)
    return len(english_words) + len(chinese_chars) + len(other_chars)


def int_to_chinese(num: int) -> str:
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    if num == 0:
        return digits[0]
    if num == 10:
        return "十"
    if num < 10:
        return digits[num]
    if num < 100:
        tens, ones = divmod(num, 10)
        prefix = "" if tens == 1 else digits[tens]
        return prefix + "十" + (digits[ones] if ones else "")

    result = []
    zero_pending = False
    chars = list(map(int, str(num)))
    length = len(chars)
    for i, digit in enumerate(chars):
        pos = length - i - 1
        if digit == 0:
            zero_pending = True
            continue
        if zero_pending and result:
            result.append("零")
        zero_pending = False
        result.append(digits[digit] + units[pos])
    return "".join(result)


def unique_token_sequences(sequences: Iterable[Sequence[int]]) -> List[List[int]]:
    seen = set()
    unique = []
    for seq in sequences:
        seq_tuple = tuple(int(x) for x in seq if int(x) >= 0)
        if not seq_tuple or seq_tuple in seen:
            continue
        seen.add(seq_tuple)
        unique.append(list(seq_tuple))
    return unique


def encode_variants(tokenizer, texts: Iterable[str]) -> List[List[int]]:
    sequences = []
    for text in texts:
        if not text:
            continue
        sequences.append(tokenizer.encode(text, add_special_tokens=False))
        if text.startswith("\n"):
            sequences.append(tokenizer.encode(text.lstrip("\n"), add_special_tokens=False))
    return unique_token_sequences(sequences)


def infer_anchor_templates(prompt: str) -> Optional[List[str]]:
    if re.search(r'"index"\s*:\s*1', prompt):
        return [',\n{{\n  "index": {n}', '\n{{\n  "index": {n}', '\n  "index": {n}']
    if "Round 1:" in prompt:
        if "## Round 1:" in prompt:
            return ["\n## Round {n}:"]
        return ["\n#*# Round {n}:", "\n## Round {n}:", "\nRound {n}:"]
    if "Floor 1:" in prompt:
        return ["\n#*# Floor {n}:", "\nFloor {n}:"]
    if "Date: Day 1:" in prompt:
        return ["\n#*# Date: Day {n}:", "\nDate: Day {n}:"]
    if "# Function 1:" in prompt:
        return ["\n# Function {n}:", "\nFunction {n}:"]
    if "% Formula 1:" in prompt:
        return ["\n% Formula {n}:", "\nFormula {n}:"]
    if "第一层" in prompt:
        return ["\n#*# 第{cn}层：", "\n#*# 第{n}层：", "\n第{cn}层：", "\n第{n}层："]
    if "日期：第1天" in prompt:
        return ["\n## 日期：第{n}天：", "\n日期：第{n}天："]
    if "# 函数1" in prompt:
        return ["\n# 函数{n}：", "\n函数{n}："]
    if "章的小说" in prompt or "chapters" in prompt.lower():
        return ["\n#*# Chapter {n}:", "\nChapter {n}:", "\n#*# 第{n}章：", "\n第{n}章："]
    return None


def render_templates(templates: Sequence[str], index: int) -> List[str]:
    rendered = []
    for template in templates:
        rendered.append(template.format(n=index, cn=int_to_chinese(index)))
    return rendered


def prompt_contains_first_anchor(prompt: str, templates: Sequence[str]) -> bool:
    for anchor in render_templates(templates, 1):
        if anchor.strip() and anchor.strip() in prompt:
            return True
    return False


def suffix_match_length(values: Sequence[int], pattern: Sequence[int]) -> int:
    max_len = min(len(values), len(pattern))
    for length in range(max_len, 0, -1):
        if list(values[-length:]) == list(pattern[:length]):
            return length
    return 0


@dataclass
class GLoBoConfig:
    total_sections: int
    section_token_target: int
    grace_tokens: int
    boost: float
    negative_bias: float
    freeform: bool
    min_new_tokens_before_eos: int
    debug: bool


class GLoBoLogitsProcessor(LogitsProcessor):
    """Stable Generation via Logits Boosting for single-sample local decoding."""

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        config: GLoBoConfig,
        anchor_templates: Optional[Sequence[str]] = None,
        initial_section_index: int = 1,
        eos_token_ids: Optional[Sequence[int]] = None,
        banned_phrases: Optional[Sequence[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.config = config
        self.anchor_templates = list(anchor_templates or [])
        self.eos_token_ids = [int(x) for x in eos_token_ids or [] if x is not None]
        self.section_index = initial_section_index if self.anchor_templates and not config.freeform else 0
        self.section_start_offset = 0
        self.last_debug_section = self.section_index

        interruption_texts = [
            ".",
            "!",
            "?",
            "。",
            "！",
            "？",
            "\n",
            "\n\n",
            "。\n",
            ".\n",
            ":\n",
            "：\n",
        ]
        self.interruption_token_ids = {
            seq[-1]
            for seq in encode_variants(tokenizer, interruption_texts)
            if seq
        }

        phrase_list = list(banned_phrases or [])
        phrase_list.extend(
            [
                "I hope this",
                "I hope these",
                "Hope this",
                "Hope these",
                "Let me know",
                "If you need",
                "如果你需要",
                "希望这些",
            ]
        )
        self.banned_sequences = encode_variants(tokenizer, phrase_list)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[0] != 1:
            raise ValueError("GLoBoLogitsProcessor currently supports batch_size=1 generation.")

        generated_ids = input_ids[0, self.prompt_length :].tolist()
        generated_len = len(generated_ids)
        self._update_section_state(generated_ids)

        self._suppress_banned_sequences(generated_ids, scores)
        self._suppress_eos_if_needed(generated_len, scores)

        if self.config.freeform or not self.anchor_templates:
            self._apply_freeform_boost(generated_ids, scores)
        else:
            self._apply_structural_boost(generated_ids, scores)

        return scores

    def _update_section_state(self, generated_ids: Sequence[int]) -> None:
        if not self.anchor_templates or self.config.freeform:
            checkpoints = len(generated_ids) // max(1, self.config.section_token_target)
            self.section_index = min(checkpoints, self.config.total_sections)
            return

        while self.section_index < self.config.total_sections:
            next_index = self.section_index + 1
            next_title_sequences = self._title_token_sequences(next_index)
            if any(len(generated_ids) >= len(seq) and list(generated_ids[-len(seq) :]) == seq for seq in next_title_sequences):
                self.section_index = next_index
                self.section_start_offset = len(generated_ids)
                if self.config.debug:
                    print(f"[GLoBo] reached section {self.section_index}")
                continue
            break

    def _title_token_sequences(self, index: int) -> List[List[int]]:
        return encode_variants(self.tokenizer, render_templates(self.anchor_templates, index))

    def _suppress_banned_sequences(self, generated_ids: Sequence[int], scores: torch.FloatTensor) -> None:
        for seq in self.banned_sequences:
            if len(seq) == 1:
                scores[0, seq[0]] += self.config.negative_bias
                continue
            prefix = seq[:-1]
            if len(generated_ids) >= len(prefix) and list(generated_ids[-len(prefix) :]) == prefix:
                scores[0, seq[-1]] += self.config.negative_bias

    def _suppress_eos_if_needed(self, generated_len: int, scores: torch.FloatTensor) -> None:
        if not self.eos_token_ids:
            return
        if self.config.freeform:
            required = max(self.config.min_new_tokens_before_eos, self.config.total_sections * self.config.section_token_target)
            should_suppress = generated_len < required
        else:
            final_section_too_short = (
                self.section_index >= self.config.total_sections
                and self._section_tokens_from_len(generated_len) < self.config.section_token_target
            )
            should_suppress = self.section_index < self.config.total_sections or final_section_too_short
        if should_suppress:
            for token_id in self.eos_token_ids:
                if token_id < scores.shape[-1]:
                    scores[0, token_id] += self.config.negative_bias

    def _section_tokens(self, generated_ids: Sequence[int]) -> int:
        return max(0, len(generated_ids) - self.section_start_offset)

    def _section_tokens_from_len(self, generated_len: int) -> int:
        return max(0, generated_len - self.section_start_offset)

    def _at_natural_interruption(self, generated_ids: Sequence[int]) -> bool:
        if not generated_ids:
            return False
        if generated_ids[-1] in self.interruption_token_ids:
            return True
        tail = self.tokenizer.decode(generated_ids[-12:], skip_special_tokens=False)
        return tail.endswith((".", "!", "?", "。", "！", "？", "\n", "\n\n", ":\n", "：\n"))

    def _condition_met(self, generated_ids: Sequence[int]) -> bool:
        section_tokens = self._section_tokens(generated_ids)
        soft = section_tokens >= self.config.section_token_target and self._at_natural_interruption(generated_ids)
        hard = section_tokens >= self.config.section_token_target + self.config.grace_tokens
        return soft or hard

    def _apply_structural_boost(self, generated_ids: Sequence[int], scores: torch.FloatTensor) -> None:
        if self.section_index >= self.config.total_sections or not self._condition_met(generated_ids):
            return

        next_index = self.section_index + 1
        title_sequences = self._title_token_sequences(next_index)
        next_tokens = set()
        for seq in title_sequences:
            match_len = suffix_match_length(generated_ids, seq)
            if match_len < len(seq):
                next_tokens.add(seq[match_len])

        for token_id in next_tokens:
            if token_id < scores.shape[-1]:
                scores[0, token_id] += self.config.boost

    def _apply_freeform_boost(self, generated_ids: Sequence[int], scores: torch.FloatTensor) -> None:
        generated_len = len(generated_ids)
        if generated_len < self.config.section_token_target:
            return
        if self._at_natural_interruption(generated_ids):
            return
        for token_id in self.interruption_token_ids:
            if token_id < scores.shape[-1]:
                scores[0, token_id] += self.config.boost * 0.5


def get_eos_token_ids(tokenizer) -> List[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return []
    if isinstance(eos, int):
        ids = [eos]
    else:
        ids = list(eos)
    for token in ("<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            ids.append(token_id)
    return sorted(set(ids))


def build_chat_input(tokenizer, prompt: str, device) -> Tuple[Dict[str, torch.Tensor], int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    return model_inputs, model_inputs.input_ids.shape[-1]


def generate_with_globo(
    model,
    tokenizer,
    prompt: str,
    args,
    total_sections: int,
    section_token_target: int,
) -> str:
    model_inputs, prompt_length = build_chat_input(tokenizer, prompt, model.device)
    inferred_templates = infer_anchor_templates(prompt)
    if args.anchor_template:
        anchor_templates = [args.anchor_template]
    else:
        anchor_templates = inferred_templates

    freeform = args.freeform or not anchor_templates
    if not anchor_templates and not args.freeform:
        print("[GLoBo] No structural anchor inferred; falling back to free-form adaptation.")

    config = GLoBoConfig(
        total_sections=total_sections,
        section_token_target=section_token_target,
        grace_tokens=args.grace_tokens,
        boost=args.boost,
        negative_bias=args.negative_bias,
        freeform=freeform,
        min_new_tokens_before_eos=args.min_new_tokens_before_eos,
        debug=args.debug_globo,
    )
    processor = GLoBoLogitsProcessor(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        config=config,
        anchor_templates=anchor_templates,
        initial_section_index=1 if anchor_templates and prompt_contains_first_anchor(prompt, anchor_templates) else 0,
        eos_token_ids=get_eos_token_ids(tokenizer),
        banned_phrases=args.banned_phrase,
    )

    generated = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        logits_processor=LogitsProcessorList([processor]),
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=get_eos_token_ids(tokenizer) or None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        early_stopping=False,
    )
    generated_ids = generated[:, prompt_length:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_existing_outputs(output_filename: str, word_count_filename: str) -> Tuple[dict, dict]:
    if os.path.exists(output_filename) and os.path.exists(word_count_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            output_dict = json.load(f)
        with open(word_count_filename, "r", encoding="utf-8") as f:
            word_count_dict = json.load(f)
        return output_dict, word_count_dict
    return {}, {}


def save_outputs(output_filename: str, word_count_filename: str, output_dict: dict, word_count_dict: dict) -> None:
    with open(word_count_filename, "w", encoding="utf-8") as f:
        json.dump(word_count_dict, f, ensure_ascii=False, indent=4)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)


def run_model_on_prompts(
    model,
    tokenizer,
    prompt_dict: Dict[str, Dict[str, str]],
    args,
    output_prefix: str,
    total_sections: int,
    section_token_target: int,
) -> Tuple[dict, dict]:
    word_count_filename = os.path.join(args.prompt_folder, f"{output_prefix}_word_count.json")
    output_filename = os.path.join(args.prompt_folder, f"{output_prefix}_output.json")
    output_dict, word_count_dict = load_existing_outputs(output_filename, word_count_filename)

    for folder_key, files_dict in tqdm(prompt_dict.items(), desc="Processing prompts"):
        for file_name, prompt_content in files_dict.items():
            prompt_key = f"{folder_key}/{file_name}"
            if args.only_prompt and args.only_prompt not in prompt_key:
                continue

            output_dict.setdefault(prompt_key, [""] * args.num_runs)
            word_count_dict.setdefault(prompt_key, [0] * args.num_runs)
            if len(output_dict[prompt_key]) < args.num_runs:
                output_dict[prompt_key].extend([""] * (args.num_runs - len(output_dict[prompt_key])))
                word_count_dict[prompt_key].extend([0] * (args.num_runs - len(word_count_dict[prompt_key])))

            for run in range(args.num_runs):
                if len(output_dict[prompt_key][run]) >= args.min_output_chars:
                    print(f"Skipping run {run + 1} for {prompt_key} (already valid)")
                    continue

                success = False
                attempts = 0
                while not success and attempts < args.max_attempts:
                    try:
                        print(f"Running {prompt_key}, run {run + 1}...")
                        output = generate_with_globo(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_content,
                            args=args,
                            total_sections=total_sections,
                            section_token_target=section_token_target,
                        )
                        if len(output) < args.min_output_chars:
                            raise ValueError("Output too short")

                        output_dict[prompt_key][run] = output
                        word_count = count_words(output)
                        word_count_dict[prompt_key][run] = word_count
                        success = True
                        print(f"Run {run + 1} successful. Model output {word_count} words.")
                        save_outputs(output_filename, word_count_filename, output_dict, word_count_dict)
                    except Exception as exc:
                        attempts += 1
                        print(f"Run {run + 1} failed (attempt {attempts}): {exc}")
                        if attempts >= args.max_attempts:
                            print(f"Max attempts reached for {prompt_key}, run {run + 1}.")
                            output_dict[prompt_key][run] = ""
                            word_count_dict[prompt_key][run] = 0
                            save_outputs(output_filename, word_count_filename, output_dict, word_count_dict)

    return output_dict, word_count_dict


def infer_counts_from_pkl_name(pkl_file: str, args) -> Tuple[int, int]:
    match = re.match(r"(\d+)_(\d+)_dict\.pkl$", pkl_file)
    total_sections = args.num_sections
    section_token_target = args.section_token_target
    if match:
        total_sections = total_sections or int(match.group(1))
        section_token_target = section_token_target or int(match.group(2))
    return total_sections or 5, section_token_target or 200


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VOLTBench outputs with local GLoBo decoding.")
    parser.add_argument("--prompt-folder", default="Longen_instructions/CH", help="Folder containing *_dict.pkl files.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Local or Hugging Face model path.")
    parser.add_argument("--model-shortname", default=None, help="Short name used in output files.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=150_000)
    parser.add_argument("--min-output-chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-sections", type=int, default=None, help="Overrides section count inferred from *_dict.pkl.")
    parser.add_argument("--section-token-target", type=int, default=None, help="Tau_max; defaults to the second number in *_dict.pkl.")
    parser.add_argument("--grace-tokens", type=int, default=100, help="Delta tokens after tau_max before hard enforcement.")
    parser.add_argument("--boost", type=float, default=12.0, help="Positive beta added to next anchor logits.")
    parser.add_argument("--negative-bias", type=float, default=-1.0e9, help="Bias for EOS and banned continuation tokens.")
    parser.add_argument("--anchor-template", default=None, help="Manual next-section anchor, e.g. '\\n#*# Chapter {n}:'.")
    parser.add_argument("--freeform", action="store_true", help="Use free-form GLoBo without explicit section anchors.")
    parser.add_argument("--min-new-tokens-before-eos", type=int, default=0)
    parser.add_argument("--banned-phrase", action="append", default=[], help="Extra phrase to suppress during decoding.")
    parser.add_argument("--only-pkl", default=None, help="Run only one pickle file, e.g. 5_200_dict.pkl.")
    parser.add_argument("--only-prompt", default=None, help="Run only prompt keys containing this substring.")
    parser.add_argument("--debug-globo", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    model_shortname = args.model_shortname or args.model.replace("/", "-")
    globo_shortname = f"{model_shortname}-GLoBo"

    for pkl_file in os.listdir(args.prompt_folder):
        if not pkl_file.endswith(".pkl"):
            continue
        if args.only_pkl and pkl_file != args.only_pkl:
            continue

        pkl_path = os.path.join(args.prompt_folder, pkl_file)
        with open(pkl_path, "rb") as f:
            loaded_prompt_dict = pickle.load(f)

        total_sections, section_token_target = infer_counts_from_pkl_name(pkl_file, args)
        print(
            f"Processing {pkl_file} with total_sections={total_sections}, "
            f"section_token_target={section_token_target}..."
        )
        pkl_name = os.path.splitext(pkl_file)[0]
        output_prefix = f"{pkl_name}_{globo_shortname}"
        run_model_on_prompts(
            model=model,
            tokenizer=tokenizer,
            prompt_dict=loaded_prompt_dict,
            args=args,
            output_prefix=output_prefix,
            total_sections=total_sections,
            section_token_target=section_token_target,
        )
        print(f"Finished processing {pkl_file}.")
