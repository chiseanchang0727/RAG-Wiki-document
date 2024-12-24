import os
import pandas as pd


def read_txt_data() -> pd.DataFrame:

    data_path = "data/raw_docs/"

    directory_list = [os.path.join(data_path, folder) for folder in os.listdir(data_path)]


    file_list = []
    for folder in directory_list:
        files = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if "topics" not in file
        ]

        file_list.extend(files)

    data = []

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            data.append({"file_name": os.path.basename(file_path), "file_content": content})

    df = pd.DataFrame(data)


    return df