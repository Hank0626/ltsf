import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ..models import (
    Autoformer,
    Crossformer,
    DLinear,
    ETSformer,
    FEDformer,
    FiLM,
    Informer,
    LightTS,
    MICN,
    Nonstationary_Transformer,
    PatchTST,
    Pyraformer,
    Reformer,
    TimesNet,
    Transformer,
)

from ..utils import EarlyStopping, adjust_learning_rate, metric, visual


class LTSFTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def set_config(self, config: dict) -> None:
        for key in config.keys():
            if key not in self.config:
                raise KeyError(f"Key '{key}' not found in config.")
            self.config[key] = config[key]

    def _acquire_device(self):
        if self.config["use_gpu"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.config["use_gpu"])
                if not self.config["use_multi_gpu"]
                else self.config["devices"]
            )
            device = torch.device(f'cuda:{self.config["gpu"]}')
            print(f'Use GPU: cuda:{self.config["gpu"]}')
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model_dict = {
            "Autoformer": Autoformer,
            "Crossformer" : Crossformer,
            "DLinear": DLinear,
            "ETSformer": ETSformer,
            "FEDformer": FEDformer,
            "FiLM": FiLM,
            "Informer": Informer,
            "LightTS": LightTS,
            "MICN": MICN,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "PatchTST": PatchTST,
            "Pyraformer": Pyraformer,
            "Reformer": Reformer,
            "TimesNet": TimesNet,
            "Transformer": Transformer,
        }

        model = model_dict[self.config["model"]](
            argparse.Namespace(**self.config)
        ).float()

        if self.config["use_multi_gpu"] and self.config["use_gpu"]:
            model = nn.DataParallel(model, device_ids=self.config["device_ids"])
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.config["pred_len"] :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [batch_y[:, : self.config["label_len"], :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.config["output_attention"]:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                else:
                    if self.config["output_attention"]:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                f_dim = -1 if self.config["features"] == "MS" else 0
                outputs = outputs[:, -self.config["pred_len"] :, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"] :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_loader, vali_loader, test_loader):
        path = os.path.join(self.config["checkpoints"])
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config["patience"], verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.config["use_amp"]:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.config["train_epochs"]):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # import pdb; pdb.set_trace()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.config["pred_len"] :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [batch_y[:, : self.config["label_len"], :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.config["output_attention"]:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                        f_dim = -1 if self.config["features"] == "MS" else 0
                        outputs = outputs[:, -self.config["pred_len"] :, f_dim:]
                        batch_y = batch_y[:, -self.config["pred_len"] :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.config["output_attention"]:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                    f_dim = -1 if self.config["features"] == "MS" else 0
                    outputs = outputs[:, -self.config["pred_len"] :, f_dim:]
                    batch_y = batch_y[:, -self.config["pred_len"] :, f_dim:].to(
                        self.device
                    )
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.config["train_epochs"] - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.config["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(
                model_optim, epoch + 1, argparse.Namespace(**self.config)
            )

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

    def test(self, setting, test=0):
        test_loader = None
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"

        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.config["pred_len"] :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [batch_y[:, : self.config["label_len"], :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.config["use_amp"]:
                    with torch.cuda.amp.autocast():
                        if self.config["output_attention"]:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                else:
                    if self.config["output_attention"]:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]

                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                f_dim = -1 if self.config["features"] == "MS" else 0
                outputs = outputs[:, -self.config["pred_len"] :, f_dim:]
                batch_y = batch_y[:, -self.config["pred_len"] :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        f = open("result.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
