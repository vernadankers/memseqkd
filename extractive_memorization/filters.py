import fasttext
import os
import sys


lid_model = fasttext.load_model("lid.176.bin")

def read_tsv(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

def basic_filter(line, srclang, trglang):
    columns = line.strip().split('\t')
    if len(columns) != 2:
        return False
    
    source, target = columns
    source_words = source.split()
    target_words = target.split()
    
    source_length = len(source_words)
    target_length = len(target_words)
    
    # too short case
    if source_length < 4:
        return False
    
    rev_ratio = target_length / source_length if source_length != 0 else 0
    ratio = source_length / target_length if target_length != 0 else 0

    src_lang_id = lid_model.predict([source])[0][0][0].split("__")[2]
    trg_lang_id = lid_model.predict([target])[0][0][0].split("__")[2]

    # spurious cases
    if ratio > 1.3 or rev_ratio > 1.3:
        return False

    # pure copy case
    if source == target:
        return False
   
    # wrong language case
    if not (src_lang_id == srclang and trg_lang_id == trglang):
        return False

    return True

def main(tsv_file_path):
    lines = read_tsv(tsv_file_path)
    
    for line in lines:
        result = basic_filter(line)
        if result:
            print(line.strip())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python basic_filter.py <tsv_file>")
    else:
        tsv_file_path = sys.argv[1]
        main(tsv_file_path)

