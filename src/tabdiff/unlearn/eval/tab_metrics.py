from copy import deepcopy
from typing import Dict, Tuple, Optional, List

import pandas as pd
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.single_table import LogisticDetection

from tabdiff.datasets import TabDiffDataset
from tabdiff.eval.metrics import reorder
from tabdiff.unlearn.eval.templates import ULMetricBase
from tabdiff.utils import remove_encoded_columns, remove_metadata_column


def remove_data_columns(
    df: pd.DataFrame,
    metadata: Dict,
    cols_to_remove: List[int],
    inplace: bool = True,
):
    """

    Notes:
        This method udpates the encoded data and meta-data objects.

    Args:
        df (): DataFrame to be updated.
        metadata (Dict): Dataset meta-data to be updated.
        cols_to_remove (List[int]): Column indices to remove (in data-space).
        inplace (bool): Unused, but indicates inplace-ness of operation.

    Returns:
        pd.DataFrame: Updated DataFrame with columns excluded.
        Dict: Updated meta-data object with columns excluded.
    """
    keep_columns = []
    for i, col_name in enumerate(df.columns):
        if not (i in cols_to_remove):
            keep_columns.append(col_name)

    ret_df = df[keep_columns]
    updated_metadata = deepcopy(metadata)
    for col_to_remove_idx in cols_to_remove:
        updated_metadata = remove_metadata_column(updated_metadata, col_to_remove_idx)
    return ret_df, updated_metadata


class TabULMetrics(ULMetricBase):

    def evaluate_c2st(self, syn_data, **kwargs):
        """
        Overwritten c2st for unlearning, ensuring columns to unlearn are removed.

        Notes:
            This implementation removes a (set of) column(s) from the data before evaluating. Thereby preventing
            the impact of the unlean(ed) columns to be used during c2st evaluation (Discriminative).

            As such, this allows to more closely match the expected unlearning, instead of testing whether the unlearned
            column matches the original distribution (expected result of unlearning).

            Additionally, this is only validated against 'adult' dataset, i.e., it may not be very stable / have
                un-foreseen effects.

        Args:
            syn_data (): Generated Synthetic dataset.

        Returns:
            Dict: Describing c2st performance
            Dict: NOt defined.
        """
        info = deepcopy(self.info)

        # We edit the synthetic data, so make a copy
        local_syn_data = deepcopy(syn_data)

        # Retreive original dataset.
        real_data = self.get_data(split="real")

        remove_columns = self._config.unlearn.column
        # TODO: Remove 'extracted' column?
        #   We'd need to do this via pre-processing, no?

        updated_real_data, _ = remove_data_columns(
            real_data,
            metadata=deepcopy(self.info),
            cols_to_remove=list(map(int, remove_columns)),
        )
        updated_synth_data, updated_info = remove_data_columns(
            syn_data,
            metadata=deepcopy(self.info),
            cols_to_remove=list(map(int, remove_columns)),
        )
        # Step 1. if we are unlearning, we must update the dataframe

        updated_real_data.columns = range(len(updated_real_data.columns))
        updated_synth_data.columns = range(len(updated_synth_data.columns))

        # Get meta-data
        metadata = updated_info["metadata"]

        # Re-order data
        metadata["columns"] = {
            int(key): value for key, value in metadata["columns"].items()
        }

        new_real_data, new_syn_data, metadata = reorder(
            updated_real_data,
            updated_synth_data,
            updated_info,
        )

        score = LogisticDetection.compute(
            real_data=new_real_data, synthetic_data=new_syn_data, metadata=metadata
        )

        out_metrics = {
            "c2st": score,
        }
        out_extras = {}
        return out_metrics, out_extras

    def evaluate_correlational_ul(
        self,
        syn_data,
        train_set: Optional[TabDiffDataset] = None,
        test_set: Optional[TabDiffDataset] = None,
        validation_set: Optional[TabDiffDataset] = None,
    ) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
        """
        Helper method to compute the correlational UL metric for a dataset, based on a target column.

        Computes utility/derived metrics of the retain columns (all those that are not targeted), and the forget
        column (those that are targeted by the unlearning scheme). Both in terms of Shape and Correlation score.

        Notes:
            Implementation assumes that vertical unlearning is performed, i.e., all features of the column are unlearned
            as a gross average.
        Args:
            syn_data (pd.DataFrame): Dataframe containing a generated sample of the datasets.

        Returns:
            out_data (Dict[str, float]): Dictionary containing Trend+Shape scores for forget and retain set.

            out_extras (Dict[str, float]): Dictionary containing the detailed scores for forget and retain set.
        """
        diag_report: DiagnosticReport
        qual_report: QualityReport
        diag_report, qual_report = self.sde_report(syn_data)

        qual_report.get_details(property_name="Column Pair Trends")
        target_col_or_cols = self._config.unlearn.column
        # TODO: Ensure multiple columns can be tracked in a single go.
        if str(target_col_or_cols).isdigit():
            target_column = int(target_col_or_cols)
            # TODO: Implement computation and caching of original dataset performance
            corr_df = qual_report.get_details(property_name="Column Pair Trends")
            density_forget_index = (corr_df["Column 1"] == target_column) | (
                corr_df["Column 2"] == target_column
            )
            density_retain_index = ~density_forget_index

            # TODO: Compute correlational score w.r.t. initial models' performance.
            # I.e., as a relative improvement / descrease in perf. as a result of training.
            density_forget_details = corr_df[density_forget_index]
            density_retain_details = corr_df[density_retain_index]

            density_forget_score = density_forget_details["Score"].mean()
            density_retain_score = density_retain_details["Score"].mean()

            shap_df = qual_report.get_details(property_name="Column Shapes")
            shap_forget_index = shap_df["Column"] == target_column
            shap_retain_index = ~shap_forget_index

            shap_forget_details = shap_df[shap_forget_index]
            shap_retain_details = shap_df[shap_retain_index]

            shap_forget_score = shap_forget_details["Score"].mean()
            shap_retain_score = shap_retain_details["Score"].mean()

            out_metrics = {
                "density/forget/Trend": density_forget_score,
                "density/retain/Trend": density_retain_score,
                "density/forget/Shape": shap_forget_score,
                "density/retain/Shape": shap_retain_score,
            }

            out_extras = {
                "forget_trends": density_forget_details,
                "retain_trends": density_retain_details,
                "forget_shape": shap_forget_details,
                "retain_shape": shap_retain_details,
            }

        else:
            raise ValueError("Multi-column unlearning is not implemented")

        return out_metrics, out_extras
