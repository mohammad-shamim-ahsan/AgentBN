import os

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
    
def clear_files(file_list):
    for filename in file_list:
        if os.path.exists(filename):
            open(filename, "w").close()
            print(f"Cleared: {filename}")