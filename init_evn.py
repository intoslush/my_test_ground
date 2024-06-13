import subprocess

def execute_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

def main():
    # 创建并克隆 base 环境，然后在 pyt 环境中安装 d2l 和 jittor 包
    print("Creating 'pyt' environment from 'base' and installing d2l and jittor...")
    command = """
    conda create -y -n pyt --clone base && \
    conda run -n pyt pip install d2l jittor psutil GPUtil ipywidgets IPython
    """
    stdout, stderr = execute_command(command)
    print(stdout)
    print(stderr)

if __name__ == "__main__":
    main()