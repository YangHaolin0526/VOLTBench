import pickle
import os
import re
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_shortname = 'Qwen2.5-1.5B'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def count_words(text):
    # 移除所有空格
    text_without_spaces = text.replace(" ", "")
    
    # 统计英文单词（连续的字母序列）
    english_words = re.findall(r'\b[a-zA-Z]+\b', text_without_spaces)
    english_word_count = len(english_words)
    
    # 统计中文汉字
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_without_spaces)
    chinese_char_count = len(chinese_chars)
    
    # 统计其他字符（非英文单词和中文汉字的部分，且不包括空格）
    other_chars = re.sub(r'\b[a-zA-Z]+\b|[\u4e00-\u9fff]', '', text_without_spaces)
    other_char_count = len(other_chars)
    
    # 总字符数 = 英文单词数 + 汉字数 + 其他字符数
    total_count = english_word_count + chinese_char_count + other_char_count
    
    return total_count

# 假设这是你的模型生成函数
def ModelGen(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in generating long-form text."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2_000_000,        # 设置理论最大值
        #min_new_tokens=500,             # 强制最小生成长度
        eos_token_id=None,                # 禁用结束符检测
        pad_token_id=tokenizer.eos_token_id,
        
        # 生成策略
        do_sample=True,                   # 启用随机采样
        temperature=0.9,                  # 提高随机性
        top_k=10,                         # 扩大候选词范围
        top_p=0.95,                       # 覆盖更多概率分布
        repetition_penalty=1.1,           # 适当抑制重复
        # 停止机制
        early_stopping=False,             # 禁用提前停止
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 主逻辑
def run_model_on_prompts(prompt_dict, num_runs=5, output_prefix="output", prompt_folder="Q1"):
    output_dict = {}  # 存储所有输出
    word_count_dict = {}  # 存储所有输出的字数

    for folder_key, files_dict in tqdm(prompt_dict.items(), desc="Processing prompts"):
        for file_name, prompt_content in files_dict.items():
            # 构造唯一的键（例如 "GenContent/Complex/file1.txt"）
            prompt_key = f"{folder_key}/{file_name}"
            output_dict[prompt_key] = []  # 初始化一个空列表存储输出
            word_count_dict[prompt_key] = []  # 初始化一个空列表存储字数

            # 运行模型 num_runs 次
            for run in range(num_runs):
                success = False
                attempts = 0
                max_attempts = 5  # 最大尝试次数

                while not success and attempts < max_attempts:
                    try:
                        print(f"Running {prompt_key}, run {run + 1}...")
                        output = ModelGen(prompt_content)  # 调用模型
                        # 检查输出长度
                        if len(output) < 200:
                            raise ValueError("Output too short")

                        output_dict[prompt_key].append(output)  # 存储输出

                        
                        word_count = count_words(output)
                        
                        word_count_dict[prompt_key].append(word_count)

                        success = True
                        print(f"Run {run + 1} successful. Model output {word_count} words.")
                        
                        # 动态生成保存文件名
                        word_count_filename = f"{prompt_folder}/{output_prefix}_word_count.json"
                        output_filename = f"{prompt_folder}/{output_prefix}_output.json"

                        with open(word_count_filename, 'w', encoding='utf-8') as f:
                            json.dump(word_count_dict, f, ensure_ascii=False, indent=4)

                        with open(output_filename, 'w', encoding='utf-8') as f:
                            json.dump(output_dict, f, ensure_ascii=False, indent=4)
                    except Exception as e:
                        attempts += 1
                        print(f"Run {run + 1} failed (attempt {attempts}): {e}")
                        if attempts >= max_attempts:
                            print(f"Max attempts reached for {prompt_key}, run {run + 1}.")
                            output_dict[prompt_key].append("")  # 存储空字符串表示失败
                            word_count_dict[prompt_key].append(0)  # 存储0表示失败

    return output_dict, word_count_dict

# 遍历文件夹中的所有 .pkl 文件并执行模型
prompt_folder = "Longen_instructions/CH"
for pkl_file in os.listdir(prompt_folder):
    if pkl_file.endswith(".pkl"):
        pkl_path = os.path.join(prompt_folder, pkl_file)
        with open(pkl_path, 'rb') as f:
            loaded_prompt_dict = pickle.load(f)
        
        print(f"Processing {pkl_file}...")
        
        # 生成输出文件名前缀
        pkl_name = os.path.splitext(pkl_file)[0]  # 去掉 .pkl 后缀
        output_prefix = f"{pkl_name}_{model_shortname}"
        
        # 运行模型并获取输出
        output_dict, word_count_dict = run_model_on_prompts(loaded_prompt_dict, num_runs=5, output_prefix=output_prefix, prompt_folder=prompt_folder)
        
        print(f"Finished processing {pkl_file}.")
