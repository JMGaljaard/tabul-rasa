import logging
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
from io import BytesIO
from typing import Dict, Any, Optional

import numpy as np
import omegaconf
import pandas as pd
import torch
from PIL import Image
from plotly import io as pio
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.single_table import LogisticDetection
from sdmetrics.visualization import get_column_plot
from sklearn.preprocessing import OneHotEncoder
from tabdiff.eval.mle.mle import get_evaluator
from tqdm import tqdm

from .template import TabMetricBase
from ..datasets import TabDiffDataset


class TabMetrics(TabMetricBase):
    def __init__(
        self,
        real_data_path,
        test_data_path,
        val_data_path,
        info,
        device,
        metric_list,
        config: omegaconf.DictConfig,
    ) -> None:
        """

        Args:
            real_data_path ():
            test_data_path ():
            val_data_path ():
            info ():
            device ():
            metric_list ():
            dataset_conf (): Dataset configuration object providing configuration of the pre-processing pipelin.
        """
        super().__init__(
            real_data_path,
            test_data_path,
            val_data_path,
            info,
            device,
            config,
        )
        self.real_data_path = real_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.info = info
        self.device = device
        self.real_data_size = len(pd.read_csv(real_data_path))
        self.metric_list = metric_list

    def evaluate(
        self,
        syn_data,
        *args,
        train_set: Optional[TabDiffDataset],
        test_set: Optional[TabDiffDataset],
        validation_set: Optional[TabDiffDataset],
        **kwargs,
    ) -> (Dict[str, Any], Dict[str, Any]):
        metrics, extras = {}, {}
        syn_data_cp = deepcopy(syn_data)
        for metric in self.metric_list:
            func = eval(f"self.evaluate_{metric}")
            print(f"Evaluating {metric}")
            # TODO: Update in case we have pre-exisitng Transformations.
            # Also provide (some optional) datasets to evaluation functions.
            out_metrics, out_extras = func(
                syn_data_cp,
                train_set=train_set,
                test_set=test_set,
                validation_set=validation_set,
            )
            metrics.update(out_metrics)
            extras.update(out_extras)
        return metrics, extras

    def evaluate_density(self, syn_data, **kwargs):
        y_only = len(syn_data.columns) == 1
        diag_report, qual_report = self.sde_report(syn_data)

        quality = qual_report.get_properties()
        diag = diag_report.get_properties()

        Shape = quality["Score"][0]
        Trend = quality["Score"][1]

        Overall = (Shape + Trend) / 2

        shape_details = qual_report.get_details(property_name="Column Shapes")
        trend_details = qual_report.get_details(property_name="Column Pair Trends")

        if y_only:
            Shape = shape_details["Score"].min()
        out_metrics = {
            "density/Shape": Shape,
            "density/Trend": Trend,
            "density/Overall": Overall,
        }
        out_extras = {"shapes": shape_details, "trends": trend_details}
        return out_metrics, out_extras

    def sde_report(
        self,
        syn_data,
        verbose=False,
        **kwargs,
    ):
        real_data = self.get_data(split="real")
        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        info = deepcopy(self.info)

        y_only = len(syn_data.columns) == 1
        if y_only:
            target_col_idx = info["target_col_idx"][0]
            syn_data = self.complete_y_only_data(syn_data, real_data, target_col_idx)
        metadata = info["metadata"]
        metadata["columns"] = {
            int(key): value for key, value in metadata["columns"].items()
        }  # ensure that keys are all integers?
        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)
        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata, verbose=verbose)
        diag_report = DiagnosticReport()
        diag_report.generate(new_real_data, new_syn_data, metadata, verbose=verbose)
        return diag_report, qual_report

    def evaluate_mle(self, syn_data, executor: ProcessPoolExecutor = None, **kwargs):
        """
        Evaluate results in parallel with provided ProcessPool.
        Args:
            syn_data ():
            executor ():
            **kwargs ():


        Returns:

        """
        train = syn_data.to_numpy()
        test = self.get_data(split="test").to_numpy()
        try:
            val = (
                pd.read_csv(self.val_data_path).to_numpy()
                if self.val_data_path
                else None
            )
        except FileNotFoundError:
            logging.warning(
                f"Tried to load {self.val_data_path} but no validation dataset available"
            )
            val = None

        if val is not None:
            logging.warning(f"Validation requested for {self.info['name']}")

        info = deepcopy(self.info)

        task_type = info["task_type"]

        evaluator = get_evaluator(task_type)

        if task_type == "regression":
            best_r2_scores, best_rmse_scores = evaluator(
                train, test, info, val=val, executor=executor
            )

            overall_scores = {}
            for score_name in ["best_r2_scores", "best_rmse_scores"]:
                overall_scores[score_name] = {}

                scores = eval(score_name)
                for method in scores:
                    name = method["name"]
                    method.pop("name")
                    overall_scores[score_name][name] = method

        else:
            (
                best_f1_scores,
                best_weighted_scores,
                best_auroc_scores,
                best_acc_scores,
                best_avg_scores,
            ) = evaluator(train, test, info, val=val, executor=executor)

            overall_scores = {}
            for score_name in [
                "best_f1_scores",
                "best_weighted_scores",
                "best_auroc_scores",
                "best_acc_scores",
                "best_avg_scores",
            ]:
                overall_scores[score_name] = {}

                scores = eval(score_name)
                for method in scores:
                    name = method["name"]
                    method.pop("name")
                    overall_scores[score_name][name] = method

        mle_score = (
            overall_scores["best_rmse_scores"]["XGBRegressor"]["RMSE"]
            if task_type == "regression"
            else overall_scores["best_auroc_scores"]["XGBClassifier"]["roc_auc"]
        )
        out_metrics = {
            "mle": mle_score,
        }
        out_extras = {
            "mle": overall_scores,
        }
        return out_metrics, out_extras

    def evaluate_c2st(self, syn_data, **kwargs):
        info = deepcopy(self.info)

        # We edit the synthetic data, so make a copy
        local_syn_data = deepcopy(syn_data)

        # Retreive original dataset.
        real_data = self.get_data(split="real")

        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        # Get meta-data
        metadata = info["metadata"]

        # Re-order data and map to
        metadata["columns"] = {
            int(key): value for key, value in metadata["columns"].items()
        }

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        score = LogisticDetection.compute(
            real_data=new_real_data, synthetic_data=new_syn_data, metadata=metadata
        )

        out_metrics = {
            "c2st": score,
        }
        out_extras = {}
        return out_metrics, out_extras

    def evaluate_dcr(self, syn_data, **kwargs):
        info = deepcopy(self.info)
        real_data = self.get_data(split="real")
        test_data = self.get_data(split="test")

        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]

        task_type = info["task_type"]
        if task_type == "regression":
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        num_ranges = []

        real_data.columns = list(np.arange(len(real_data.columns)))
        syn_data.columns = list(np.arange(len(real_data.columns)))
        test_data.columns = list(np.arange(len(real_data.columns)))
        for i in num_col_idx:
            num_ranges.append(real_data[i].max() - real_data[i].min())

        num_ranges = np.array(num_ranges)

        num_real_data = real_data[num_col_idx]
        cat_real_data = real_data[cat_col_idx]
        num_syn_data = syn_data[num_col_idx]
        cat_syn_data = syn_data[cat_col_idx]
        num_test_data = test_data[num_col_idx]
        cat_test_data = test_data[cat_col_idx]

        num_real_data_np = num_real_data.to_numpy()
        cat_real_data_np = cat_real_data.to_numpy().astype("str")
        num_syn_data_np = num_syn_data.to_numpy()
        cat_syn_data_np = cat_syn_data.to_numpy().astype("str")
        num_test_data_np = num_test_data.to_numpy()
        cat_test_data_np = cat_test_data.to_numpy().astype("str")

        encoder = OneHotEncoder()
        cat_complete_data_np = np.concatenate(
            [cat_real_data_np, cat_test_data_np], axis=0
        )
        encoder.fit(cat_complete_data_np)
        # encoder.fit(cat_real_data_np)

        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
        cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()

        num_real_data_np = num_real_data_np / num_ranges
        num_syn_data_np = num_syn_data_np / num_ranges
        num_test_data_np = num_test_data_np / num_ranges

        real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
        syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
        test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

        device = self.device

        real_data_th = torch.tensor(real_data_np).to(device)
        syn_data_th = torch.tensor(syn_data_np).to(device)
        test_data_th = torch.tensor(test_data_np).to(device)

        dcrs_real = []
        dcrs_test = []
        batch_size = (
            10000 // cat_real_data_oh.shape[1]
        )  # This esitmation should make sure that dcr_real and dcr_test can be fit into 10GB GPU memory

        for i in tqdm(range((syn_data_th.shape[0] // batch_size) + 1)):
            if i != (syn_data_th.shape[0] // batch_size):
                batch_syn_data_th = syn_data_th[i * batch_size : (i + 1) * batch_size]
            else:
                batch_syn_data_th = syn_data_th[i * batch_size :]

            dcr_real = (
                (batch_syn_data_th[:, None] - real_data_th)
                .abs()
                .sum(dim=2)
                .min(dim=1)
                .values
            )
            dcr_test = (
                (batch_syn_data_th[:, None] - test_data_th)
                .abs()
                .sum(dim=2)
                .min(dim=1)
                .values
            )
            dcrs_real.append(dcr_real)
            dcrs_test.append(dcr_test)

        dcrs_real = torch.cat(dcrs_real)
        dcrs_test = torch.cat(dcrs_test)

        score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]

        out_metrics = {
            "dcr": score,
        }
        out_extras = {
            "dcr_real": dcrs_real.cpu().numpy(),
            "dcr_test": dcrs_test.cpu().numpy(),
        }
        return out_metrics, out_extras

    def plot_density(self, syn_data, **kwargs):
        syn_data_cp = deepcopy(syn_data)
        real_data = self.get_data(split="real")

        info = deepcopy(self.info)
        y_only = len(syn_data_cp.columns) == 1
        if y_only:
            target_col_idx = info["target_col_idx"][0]
            target_col_name = info["column_names"][target_col_idx]
            syn_data_cp = self.complete_y_only_data(
                syn_data_cp, real_data, target_col_name
            )
        img = plot_density(syn_data_cp, real_data, info)
        return img

    def complete_y_only_data(self, syn_data, real_data, target_col_idx):
        syn_target_col = deepcopy(syn_data.iloc[:, 0])
        syn_data = deepcopy(real_data)
        syn_data[target_col_idx] = syn_target_col
        return syn_data


def reorder(real_data, syn_data, info):
    num_col_idx = deepcopy(
        info["num_col_idx"]
    )  # BUG: info will be modified by += in the next few lines
    cat_col_idx = deepcopy(info["cat_col_idx"])
    target_col_idx = deepcopy(info["target_col_idx"])

    task_type = info["task_type"]
    if task_type == "regression":
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]

    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    metadata = info["metadata"]

    columns = metadata["columns"]
    metadata["columns"] = {}

    inverse_idx_mapping = info["inverse_idx_mapping"]

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata["columns"][i] = columns[num_col_idx[i]]
        else:
            metadata["columns"][i] = columns[cat_col_idx[i - len(num_col_idx)]]

    return new_real_data, new_syn_data, metadata


def plot_density(syn_data, real_data, info, num_per_row=3):
    column_names = info["column_names"]
    num_cat = len(column_names)
    num_col = num_per_row
    num_row = (num_cat - 1) // num_col + 1

    imgs = []
    for i, col in tqdm(enumerate(column_names), total=len(column_names)):
        # plot_type = 'bar' if i in info['cat_col_idx'] else 'distplot'
        plot_type = (
            "bar"
            if info["metadata"]["columns"][str(i)]["sdtype"] == "categorical"
            else "distplot"
        )
        if (
            plot_type == "distplot" and (syn_data[col][0] == syn_data[col]).all()
        ):  # to tackle a very weird bug
            # If the continuous data all aggregate at a single value, get_column_plot() cannot plot a density curve for it.
            # So, we perturb one entry of the cont data by a small amount
            print(
                f"\n ALERT: the generated samples column_{i} with name '{col}' all has the same value of {syn_data[col][0]} \n"
            )
            syn_data[col][0] += 1e-5
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=syn_data,
            column_name=col,
            plot_type=plot_type,
        )

        img_bytes = pio.to_image(fig, format="png")
        img = Image.open(BytesIO(img_bytes))
        imgs.append(img)

    width, height = imgs[0].size
    big_img = Image.new("RGB", (width * num_col, height * num_row))
    for i, img in enumerate(imgs):
        coordinate = (i % num_col * width, i // num_col * height)
        big_img.paste(img, coordinate)
    return big_img
