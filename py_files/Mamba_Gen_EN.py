from transformers import pipeline
import pickle
import os
import re
import json
from tqdm import tqdm

# 指定可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 模型路径
model_name = "tiiuae/Falcon3-Mamba-7B-Instruct"
#model_name = "4omini"
model_shortname = "Falcon3-Mamba-7B"
# 使用 pipeline
pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto"  # 启用多 GPU 支持
)

def ModelGen(prompt):
    # 构造输入
    messages = [
        {"role": "user", "content": prompt},
    ]

    # 生成文本
    response = pipe(
        messages,
        #max_length=4096,  # 设置为模型支持的最大上下文长度
        max_new_tokens=2000000,  # 控制生成的新 token 数量
        temperature=0.9,  # 增加生成文本的多样性
        do_sample=True,  # 启用采样模式
        eos_token_id=None,  # 禁用提前停止
        early_stopping=False,  # 禁用提前停止
    )

    output = response[0]['generated_text'][1]['content']
    #print('output:', output)
    print('------------------')
    return output

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

# 主逻辑
skip_key = ["GenContent/Simple/Xiyouji.txt", "GenContent/Simple/Harry.txt", "GenContent/Complex/Xiyouji.txt", "GenContent/Complex/Harry.txt", ]
def run_model_on_prompts(prompt_dict, num_runs=5, output_prefix="output", prompt_folder="Q1"):
    output_dict = {}  # 存储所有输出
    word_count_dict = {}  # 存储所有输出的字数

    # 动态生成保存文件名
    word_count_filename = f"{prompt_folder}/{output_prefix}_word_count.json"
    output_filename = f"{prompt_folder}/{output_prefix}_output.json"

    # 检查是否已经存在输出文件
    if os.path.exists(output_filename) and os.path.exists(word_count_filename):
        print('Existing output and word count files...')
        with open(output_filename, 'r', encoding='utf-8') as f:
            output_dict = json.load(f)
        with open(word_count_filename, 'r', encoding='utf-8') as f:
            word_count_dict = json.load(f)
    else:
        print('Not Existing output and word count files...')

        output_dict = {}
        word_count_dict = {}

    for folder_key, files_dict in tqdm(prompt_dict.items(), desc="Processing prompts"):
        for file_name, prompt_content in files_dict.items():
            # 构造唯一的键（例如 "GenContent/Complex/file1.txt"）
            prompt_key = f"{folder_key}/{file_name}"
            if prompt_key in skip_key:
                print(f'Skip {prompt_key}')
                continue

            # 检查是否已经处理过这个 prompt_key
            if prompt_key in output_dict:
                # 检查每一轮输出的字符数是否都超过 200
                valid_runs = [len(output) >= 200 for output in output_dict[prompt_key]]
                if all(valid_runs) and len(output_dict[prompt_key]) >= num_runs:
                    print(f'Skipping already processed {prompt_key}')
                    continue
                else:
                    # 如果有某一轮的输出字符数不足 200，则重新执行该轮
                    for run in range(len(output_dict[prompt_key])):
                        if len(output_dict[prompt_key][run]) < 200:
                            print(f'Re-running {prompt_key}, run {run + 1} due to insufficient output length')
                            output_dict[prompt_key][run] = ""  # 清空该轮结果
                            word_count_dict[prompt_key][run] = 0  # 清空该轮字数
            else:
                # 如果 prompt_key 不存在，初始化空列表
                output_dict[prompt_key] = [""] * num_runs
                word_count_dict[prompt_key] = [0] * num_runs

            # 运行模型 num_runs 次
            for run in range(num_runs):
                # 如果当前轮已经有有效结果（字符数 >= 200），跳过
                if len(output_dict[prompt_key][run]) >= 200:
                    print(f'Skipping run {run + 1} for {prompt_key} (already valid)')
                    continue

                success = False
                attempts = 0
                max_attempts = 5  # 最大尝试次数

                while not success and attempts < max_attempts:
                    try:
                        print(f"Running {prompt_key}, run {run + 1}...")
                        all_output = ModelGen(prompt_content)  # 调用模型
                        output = all_output
                        # 检查输出长度
                        if len(output) < 200:
                            raise ValueError("Output too short")

                        output_dict[prompt_key][run] = all_output  # 更新该轮输出
                        word_count = count_words(output)
                        word_count_dict[prompt_key][run] = word_count  # 更新该轮字数

                        success = True
                        print(f"Run {run + 1} successful. Model output {word_count} words.")

                        # 每次成功运行后保存结果
                        with open(word_count_filename, 'w', encoding='utf-8') as f:
                            json.dump(word_count_dict, f, ensure_ascii=False, indent=4)

                        with open(output_filename, 'w', encoding='utf-8') as f:
                            json.dump(output_dict, f, ensure_ascii=False, indent=4)
                    except Exception as e:
                        attempts += 1
                        print(f"Run {run + 1} failed (attempt {attempts}): {e}")
                        if attempts >= max_attempts:
                            print(f"Max attempts reached for {prompt_key}, run {run + 1}.")
                            output_dict[prompt_key][run] = ""  # 存储空字符串表示失败
                            word_count_dict[prompt_key][run] = 0  # 存储0表示失败

    return output_dict, word_count_dict

# 遍历文件夹中的所有 .pkl 文件并执行模型
skip_pkl = ['500_200_dict.pkl']
#skip_pkl = ['5_200_dict.pkl', '20_200_dict.pkl', '50_200_dict.pkl', '100_200_dict.pkl']
#test_pkl = ['10_200_dict.pkl']
prompt_folder = "Longen_instructions/EN"
for pkl_file in os.listdir(prompt_folder):
    if pkl_file.endswith(".pkl"):
        pkl_path = os.path.join(prompt_folder, pkl_file)
        if pkl_file in skip_pkl:
            print(f'Skip {pkl_file}...')
            continue
        
        with open(pkl_path, 'rb') as f:
            loaded_prompt_dict = pickle.load(f)
        
        print(f"Processing {pkl_file}...")
        
        # 生成输出文件名前缀
        pkl_name = os.path.splitext(pkl_file)[0]  # 去掉 .pkl 后缀
        output_prefix = f"{pkl_name}_{model_shortname}"
        
        # 运行模型并获取输出
        output_dict, word_count_dict = run_model_on_prompts(loaded_prompt_dict, num_runs=5, output_prefix=output_prefix, prompt_folder=prompt_folder)
        
        print(f"Finished processing {pkl_file}.")
