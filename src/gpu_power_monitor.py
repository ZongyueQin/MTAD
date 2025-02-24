import subprocess
import numpy as np
import time

def get_power():
    data = subprocess.run("nvidia-smi --query-gpu=power.draw --format=csv", capture_output=True, shell=True, text=True)
    out = data.stdout.split('\n')[1:]
    out = [float(x[:-1]) for x in out if x != '']
    return np.sum(out)

def main():
    while True:
        print(time.time(), get_power())
        time.sleep(1)

if __name__ == '__main__':
    main()
