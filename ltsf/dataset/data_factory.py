import os
import shutil
import zipfile
from os.path import expanduser

import requests
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
)


class DatasetFactory:
    @staticmethod
    def create(
        base_class_name, download=False, dataset_path=expanduser("~"), config=None
    ):
        base_classes = {
            "ETTh": Dataset_ETT_hour,
            "ETTm": Dataset_ETT_minute,
            "ECL": Dataset_Custom,
            "ILI": Dataset_Custom,
            "Traffic": Dataset_Custom,
            "Weather": Dataset_Custom,
            "Exchange-Rate": Dataset_Custom,
        }

        if download and not os.path.exists(
            os.path.join(expanduser(dataset_path), "all_six_datasets.zip")
        ):
            DatasetFactory.download_zip(dataset_path)

        base_class = base_classes.get(base_class_name, None)
        if not base_class:
            raise ValueError(f"No base class found for name: {base_class_name}")

        train_dataset = DatasetFactory.get_dataset(
            base_class, "train", dataset_path, config
        )
        val_dataset = DatasetFactory.get_dataset(
            base_class, "val", dataset_path, config
        )
        test_dataset = DatasetFactory.get_dataset(
            base_class, "test", dataset_path, config
        )

        train_loader = DatasetFactory.get_dataloader(train_dataset, config, "train")
        val_loader = DatasetFactory.get_dataloader(val_dataset, config, "val")
        test_loader = DatasetFactory.get_dataloader(test_dataset, config, "test")

        return train_loader, val_loader, test_loader

    @staticmethod
    def download_zip(root_dir):
        response = requests.get(
            "https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/files/?p=%2Fall_six_datasets.zip&dl=1",
            stream=True,
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to download dataset: {response.status_code}")

        total_size = int(response.headers.get("content-length", 0))

        zip_file_path = os.path.join(expanduser(root_dir), "all_six_datasets.zip")
        with open(zip_file_path, "wb") as f:
            for data in tqdm(
                response.iter_content(1024**2),
                total=total_size / 1024**2,
                unit="MB",
                unit_scale=True,
                colour="green",
            ):
                f.write(data)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(expanduser(root_dir))

        extracted_folder = os.path.join(expanduser(root_dir), zip_ref.namelist()[0])
        renamed_folder = os.path.join(expanduser(root_dir), "dataset")
        shutil.move(extracted_folder, renamed_folder)

        os.remove(zip_file_path)
        print("Download and unzip successful.")

    @staticmethod
    def get_dataset(Datacls, flag, dataset_path, config):
        timeenc = 0 if config["embed"] != "timeF" else 1
        return Datacls(
            root_path=os.path.join(dataset_path, "dataset"),
            data_path=config["data_path"],
            flag=flag,
            size=[config["seq_len"], config["label_len"], config["pred_len"]],
            features=config["features"],
            target=config["target"],
            timeenc=timeenc,
            freq=config["freq"],
        )

    @classmethod
    def get_dataloader(cls, dataset, config, flag):
        if flag == "test" or flag == "pred":
            shuffle_flag = False
            drop_last = False
        else:
            shuffle_flag = True
            drop_last = True
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
        )
