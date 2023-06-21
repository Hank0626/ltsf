import os
import shutil
import zipfile
from os.path import expanduser

import requests
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute


class DatasetFactory:
    """
    A factory class for creating and handling different types of datasets, downloading dataset if required and creating
    dataloaders for training, validation, and testing. This class allows for easy addition of new dataset types and
    handles the specifics of each type.
    """

    @staticmethod
    def create(
        config: object,
        download=True,
        data_path=expanduser("~"),
    ):
        """
        Creates dataloaders for training, validation, and testing based on the given configuration.

        Args:
            config (object): The configuration object containing dataset type and model parameters.
            download (bool): Whether to download the dataset or not. Default is True.
            data_path (str): The path where dataset is located or will be downloaded. Default is user's home directory.

        Returns:
            tuple: A tuple containing training, validation, and testing dataloaders.
        """
        config = config.config_dict
        if download and not os.path.exists(
            os.path.join(expanduser(data_path), "dataset")
        ):
            print("\033[34m" + "Start downloading dataset..." + "\033[0m")
            DatasetFactory.download_zip(data_path)

        data_name = config["data"]
        base_classes = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "ECL": Dataset_Custom,
            "ILI": Dataset_Custom,
            "Traffic": Dataset_Custom,
            "Weather": Dataset_Custom,
            "Exchange-Rate": Dataset_Custom,
        }

        print(
            "\033[34m" + "Loading dataset: " + "\033[32m" + f"{data_name}" + "\033[0m"
        )

        base_class = base_classes.get(data_name, None)
        if not base_class:
            raise ValueError(f"No base class found for name: {data_name}")

        train_dataset = DatasetFactory.get_dataset(
            base_class, "train", data_path, config
        )
        val_dataset = DatasetFactory.get_dataset(base_class, "val", data_path, config)
        test_dataset = DatasetFactory.get_dataset(base_class, "test", data_path, config)
        print(
            f"\t train_dataset: {len(train_dataset)}\n\t val_dataset: {len(val_dataset)}\n\t test_dataset: {len(test_dataset)}\n"
        )
        train_loader = DatasetFactory.get_dataloader(train_dataset, config, "train")
        val_loader = DatasetFactory.get_dataloader(val_dataset, config, "val")
        test_loader = DatasetFactory.get_dataloader(test_dataset, config, "test")
        return train_loader, val_loader, test_loader

    @staticmethod
    def download_zip(root_dir):
        """
        Downloads and unzips the dataset from a given URL into the specified directory.

        Args:
            root_dir (str): The directory where the dataset will be downloaded.

        Raises:
            ValueError: If the status code of the response is not 200.
        """
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
    def get_dataset(Datacls, flag, data_path, config):
        """
        Creates a dataset of the given class based on the specified flag.

        Args:
            Datacls (class): The class of the dataset to be created.
            flag (str): The flag indicating whether the dataset is for training, validation or testing.
            dataset_path (str): The path to the dataset.
            config (object): The configuration object containing dataset type and model parameters.

        Returns:
            object: An instance of the specified dataset class.
        """
        timeenc = 0 if config["embed"] != "timeF" else 1
        return Datacls(
            root_path=os.path.join(data_path, "dataset"),
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
        """
        Creates a dataloader for the given dataset.

        Args:
            dataset (object): The dataset for which dataloader is to be created.
            config (object): The configuration object containing dataset type and model parameters.
            flag (str): The flag indicating whether the dataloader is for training, validation or testing.

        Returns:
            DataLoader: A dataloader for the specified dataset.
        """
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
