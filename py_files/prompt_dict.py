import argparse
import os
import pickle

def load_prompts_to_dict(folder_path, num_section, word_section):
    prompt_dict = {}

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                # 获取文件的相对路径
                relative_path = os.path.relpath(root, folder_path)
                # 构造字典的键，例如 "GenContent/Complex" 或 "GenData/Simple"
                key = relative_path.replace(os.sep, '/')
                # 读取文件内容
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查并替换 {num_section}
                if "{num_section}" in content:
                    content = content.replace("{num_section}", str(num_section))
                
                # 检查并替换 {word_section}
                if "{word_section}" in content:
                    content = content.replace("{word_section}", str(word_section))
                
                # 将内容存储到字典中
                if key not in prompt_dict:
                    prompt_dict[key] = {}
                prompt_dict[key][file] = content

    return prompt_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Build prompt pickle files from Longen_instructions.")
    parser.add_argument("--prompt-root", default="Longen_instructions", help="Root folder containing EN/CH prompts.")
    parser.add_argument("--language", choices=["EN", "CH"], default="EN", help="Prompt language to process.")
    parser.add_argument("--num-section", type=int, default=5, help="Value used to replace {num_section}.")
    parser.add_argument("--word-section", type=int, default=200, help="Value used to replace {word_section}.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder_path = os.path.join(args.prompt_root, args.language)

    prompt_dict = load_prompts_to_dict(folder_path, args.num_section, args.word_section)

    output_filename = f"{args.num_section}_{args.word_section}_dict.pkl"
    output_path = os.path.join(folder_path, output_filename)

    with open(output_path, 'wb') as f:
        pickle.dump(prompt_dict, f)

    print(f"prompt_dict saved to {output_path}")

    for key, value in prompt_dict.items():
        print(f"{key}:")
        for file_name, content in value.items():
            print(f"  {file_name}: {content[:50]}...")
