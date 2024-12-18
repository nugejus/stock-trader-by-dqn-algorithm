import os
import pandas

def prepare_data():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(ROOT_DIR, "archive")
    archive = ["BTC-2017min.csv", "BTC-2018min.csv", "BTC-2019min.csv", "BTC-2020min.csv", "BTC-2021min.csv"]

    for arch in archive:
        file_name = os.path.join(file_dir, arch)
        data = pandas.read_csv(file_name)
        data = data.drop(["date","unix", "symbol", "Volume USD"],axis= 1)

        yield data