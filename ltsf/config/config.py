import time


class Config:
    """
    This class handles configuration for different models and datasets. It validates the dataset type, calculates
    necessary parameters based on model type and sets up a configuration dictionary. This dictionary (`config_dict`)
    is used to set up the model parameters for different machine learning tasks.

    `config_dict` includes:
        - task_name: The name of the task.
        - is_training: Flag to indicate whether model is in training mode.
        - model: The model name.
        - data: The data type.
        - data_path: The path to the data.
        - features: The feature type.
        - target: The target.
        - freq: The frequency of data.
        - checkpoints: The location to store checkpoints.
        - seq_len: The length of the sequence.
        - label_len: The length of the label.
        - pred_len: The length of the prediction.
        - seasonal_patterns: The seasonal patterns.
        - mask_rate: The masking rate.
        - anomaly_ratio: The ratio of anomalies.
        - top_k: Top K value for certain computations.
        - num_kernels: The number of kernels for convolution operations.
        - enc_in: The number of input features for the encoder.
        - dec_in: The number of input features for the decoder.
        - c_out: The number of output features.
        - d_model: The dimensionality of the model.
        - n_heads: The number of attention heads.
        - e_layers: The number of layers in the encoder.
        - d_layers: The number of layers in the decoder.
        - d_ff: The dimensionality of the feed-forward network model.
        - moving_avg: The moving average period.
        - factor: A factor for certain computations.
        - distil: Flag indicating whether distillation is to be used.
        - dropout: The dropout rate.
        - embed: The type of embedding.
        - activation: The activation function used.
        - output_attention: Flag indicating whether output attention is to be used.
        - num_workers: The number of worker threads for loading the data.
        - itr: The iteration.
        - train_epochs: The number of training epochs.
        - batch_size: The size of the batch.
        - patience: The patience level for early stopping.
        - learning_rate: The learning rate.
        - des: Description of the model.
        - loss: The loss function.
        - lradj: The learning rate adjustment type.
        - use_amp: Flag to indicate usage of Automatic Mixed Precision.
        - use_gpu: Flag indicating whether to use GPU.
        - gpu: The GPU to use.
        - use_multi_gpu: Flag indicating whether to use multiple GPUs.
        - devices: The devices to be used.
        - p_hidden_dims: The dimensions of the hidden layers in the predictor.
        - p_hidden_layers: The number of hidden layers in the predictor.
        - chunk_size: The size of chunks for certain computations.
        - bucket_size: The size of the bucket for certain computations.
        - n_hashes: The number of hashes for certain computations.
    """

    def __init__(
        self,
        model: str,
        data: str,
        checkpoints: str = "./checkpoints",
        use_gpu: bool = True,
        print_config: bool = True,
    ):
        """
        Initializer for the Config class. Validates the data type, calculates necessary parameters and sets up the
        configuration dictionary.

        Args:
            model (str): The model type.
            data (str): The data type.
            checkpoints (str): The location to store checkpoints. Default is "./checkpoints".
            use_gpu (bool): Whether to use GPU. Default is True.
            print_config (bool): Whether to print the configuration. Default is True.
        """
        if data not in [
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "ECL",
            "ILI",
            "Traffic",
            "Weather",
            "Exchange-Rate",
        ]:
            raise ValueError(f"Not support data type {data}")

        if model not in [
            "Autoformer",
            "Crossformer",
            "DLinear",
            "ETSformer",
            "FEDformer",
            "FiLM",
            "Informer",
            "LightTS",
            "MICN",
            "Nonstationary_Transformer",
            "PatchTST",
            "Pyraformer",
            "Reformer",
            "TimesNet",
            "Transformer",
        ]:
            raise ValueError(f"Not support model type {model}")

        seq_len, label_len, pred_len = 96, 48, 96

        if data == "ECL":
            data_path = "./electricity/electricity.csv"
            enc_in = dec_in = c_out = 321
        elif data == "ILI":
            data_path = "./illness/national_illness.csv"
            seq_len, label_len, pred_len = 36, 18, 24
            enc_in = dec_in = c_out = 7
        elif data == "Traffic":
            data_path = "./traffic/traffic.csv"
            enc_in = dec_in = c_out = 862
        elif data == "Weather":
            data_path = "./weather/weather.csv"
            enc_in = dec_in = c_out = 21
        elif data == "Exchange-Rate":
            data_path = "./exchange_rate/exchange_rate.csv"
            enc_in = dec_in = c_out = 8
        elif data.startswith("ETT"):
            data_path = f"./ETT-small/{data}.csv"
            enc_in = dec_in = c_out = 7

        if model == "ETSformer":
            e_layers = d_layers = 1
        else:
            e_layers = 2
            d_layers = 1

        if model == "LightTS":
            chunk_size = seq_len // 4
        else:
            chunk_size = None

        if model == "MICN":
            label_len = seq_len

        self.config_dict = {
            "task_name": "long_term_forecast",
            "is_training": 1,
            "model": model,
            "data": data,
            "data_path": data_path,
            "features": "M",
            "target": "OT",
            "freq": "h",
            "checkpoints": checkpoints,
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "seasonal_patterns": "Monthly",
            "mask_rate": 0.25,
            "anomaly_ratio": 0.25,
            "top_k": 5,
            "num_kernels": 6,
            "enc_in": enc_in,
            "dec_in": dec_in,
            "c_out": c_out,
            "d_model": 256,
            "n_heads": 8,
            "e_layers": e_layers,
            "d_layers": d_layers,
            "d_ff": 512,
            "moving_avg": 25,
            "factor": 1,
            "distil": True,
            "dropout": 0.1,
            "embed": "timeF",
            "activation": "gelu",
            "output_attention": False,
            "num_workers": 10,
            "itr": 1,
            "train_epochs": 10,
            "batch_size": 32,
            "patience": 3,
            "learning_rate": 0.0001,
            "des": "test",
            "loss": "MSE",
            "lradj": "type1",
            "use_amp": False,
            "use_gpu": use_gpu,
            "gpu": 0,
            "use_multi_gpu": False,
            "devices": "0,1,2,3",
            "p_hidden_dims": [128, 128],
            "p_hidden_layers": 2,
            "chunk_size": chunk_size,
            "bucket_size": 4,
            "n_hashes": 4,
        }

        if print_config:
            self.print_config()
            time.sleep(5)

    def set_config(self, dic: dict):
        """
        Updates the configuration dictionary with the provided dictionary.

        Args:
            dic (dict): Dictionary of configuration parameters to update.

        Raises:
            ValueError: If a key in the provided dictionary is not found in the configuration dictionary.
        """
        for key, value in dic.items():
            if key not in self.config_dict:
                raise ValueError(f"Invalid key {key}")
            self.config_dict[key] = value

    @property
    def get_setting(self):
        """
        Generates a settings string based on the current configuration.

        Returns:
            str: The generated settings string.
        """

        setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}".format(
            self.config_dict["model"],
            self.config_dict["data"],
            self.config_dict["features"],
            self.config_dict["seq_len"],
            self.config_dict["label_len"],
            self.config_dict["pred_len"],
            self.config_dict["d_model"],
            self.config_dict["n_heads"],
            self.config_dict["e_layers"],
            self.config_dict["d_layers"],
            self.config_dict["d_ff"],
            self.config_dict["factor"],
            self.config_dict["embed"],
            self.config_dict["distil"],
            self.config_dict["des"],
        )
        return setting

    def print_config(self):
        """
        Prints the current configuration.
        """
        print("\033[36m" + "<=============Configuration============>" + "\033[0m")
        for key, value in self.config_dict.items():
            print(f"{key} : {value}")
        print("\033[36m" + "<======================================>\n\n" + "\033[0m")
