import argparse
import os
import pickle
from natsort import natsorted

import pathlib

HERE = pathlib.Path(__file__).parent

def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def analyze_pkl_file(file_path):
    data = load_pkl_file(file_path)
    print(f"Analyzing file: {file_path}")
    print(f"Keys in the pickle file: {list(data.keys())}")
    # Add more analysis as needed
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_folder", type=str, required=True)
    args = parser.parse_args()

    src_folder = args.motion_folder

    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.pkl"):
                file_path = os.path.join(dirpath, filename)
                print("skip stagei file:", file_path)
                continue
            if filename.endswith((".pkl")):
                file_path = os.path.join(dirpath, filename)
                args_list.append(file_path)

    print("file num: ", len(args_list))

if __name__ == "__main__":
    main()