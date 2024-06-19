import subprocess
import requests
import zipfile
import os

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {output_path}")

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")
def execute_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr


def main():
    # 创建并克隆 base 环境，然后在 pyt 环境中安装 d2l 和 jittor 包
    print("Creating 'pyt' environment from 'base' and installing d2l and jittor...")
    command = """
    conda create -y -n pyt --clone base && \
    conda run -n pyt pip install d2l jittor psutil GPUtil ipywidgets IPython ftfy regex tqdm
    """
    stdout, stderr = execute_command(command)
    print(stdout)
    print(stderr)
    print("接下来开始下载数据")
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    files_to_download = [
        {"url": "https://cloud.tsinghua.edu.cn/f/212edd1e7b3b44f5b301/?dl=1", "output": os.path.join(data_dir, "train.txt")},
        {"url": "https://cloud.tsinghua.edu.cn/f/418b311c5ae8484f8208/?dl=1", "output": os.path.join(data_dir, "classname.txt")},
        {"url": "https://cloud.tsinghua.edu.cn/f/7c44b138a6344f4b8fd1/?dl=1", "output": os.path.join(data_dir, "TrainSet.zip")},
        {"url": "https://cloud.tsinghua.edu.cn/f/c00ca0f3f27340899a05/?dl=1", "output": os.path.join(data_dir, "TestSetA.zip")},
        {"url": "https://github.com/uyzhang/JCLIP/releases/download/%E6%9D%83%E9%87%8D/ViT-B-32.pkl", "output": os.path.join(data_dir, "ViT-B-32.pkl")}
    ]

    for file_info in files_to_download:
        download_file(file_info["url"], file_info["output"])
        
        # If the downloaded file is a zip file, extract it and delete the zip file
        if file_info["output"].endswith(".zip"):
            unzip_file(file_info["output"], data_dir)
            os.remove(file_info["output"])
            print(f"Deleted {file_info['output']}")

if __name__ == "__main__":
    main()