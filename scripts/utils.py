import logging

import pandas as pd
import numpy as np
import prophet


def validate_input_timeseries(time_series: pd.DataFrame):
    assert time_series.index.name == "ds", "unexpected index name"
    all_numeric = all(np.issubdtype(dtype, np.number) for dtype in time_series.dtypes)
    assert all_numeric, "non-numeric data types in table"


def fix_negatives(avg_pred: pd.DataFrame) -> pd.DataFrame:
    avg_pred = avg_pred.mask(avg_pred < 0)
    avg_pred = avg_pred.dropna(how="all", axis=1)
    avg_pred.columns = range(avg_pred.shape[1])
    avg_pred = avg_pred.interpolate()
    avg_pred = avg_pred.fillna(method="bfill").fillna(method="ffill")

    avg_pred = avg_pred.dropna(axis=1)
    avg_pred = pd.concat(
        [avg_pred, avg_pred.sample(1000 - len(avg_pred.columns), replace=True, axis=1)],
        axis=1,
        ignore_index=True,
    )

    return avg_pred


def create_predictions(time_series: pd.DataFrame) -> pd.DataFrame:
    """create_predictions _summary_

    Args:
        time_series (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # drop incomplete time series
    time_series = time_series.set_index(["case", "ds"]).y.unstack("case").dropna(axis=1)

    validate_input_timeseries(time_series)

    predictions = {}

    for case, ts in time_series.items():
        # of each case
        ts = ts.rename("y").reset_index()

        m = prophet.Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(
            periods=12 * 5, freq="m", include_history=False
        )
        forecast = m.predict(future)

        avg_pred = pd.DataFrame(m.predictive_samples(future)["yhat"])
        avg_pred.index = future.ds
        avg_pred = avg_pred.resample("Y").mean()

        if (avg_pred < 0).all().all():
            # all negatives, impossible to fix
            logging.warning(f"{case} all negative")

            continue

        fixed_pred = fix_negatives(avg_pred)
        assert (fixed_pred > 0).all().all()
        assert fixed_pred.isna().sum().sum() == 0, "null values present"
        predictions[case] = fixed_pred

    predictions_df = pd.concat(predictions.values(), keys=predictions.keys(), 
                               axis=1)
    predictions_df.columns.names = ["subject", "case"]
    predictions_df = predictions_df.stack("case").swaplevel().sort_index()

    return predictions_df


def marginal_supply(predictions_df: pd.DataFrame, cutoff=0.01):
    # remove production at t0
    t_ini = predictions_df.index.get_level_values("ds")[0]
    diff_predictions = (predictions_df.unstack("case") -
    predictions_df.unstack("case").loc[t_ini]
    )  # .stack('case').swaplevel().sort_index()

    # identify cases with negative supply
    cases_decreasing_supply = (
        (diff_predictions.stack(["case", "subject"]).groupby("case").sum() < 0)
        .replace(False, np.nan)
        .dropna()
        .index
    )

    # set to 0 negative contributions
    diff_predictions[
        diff_predictions.columns[diff_predictions.sum() < 0]
    ] = 0  # do not contribute to increase dmd

    market_share = (
        diff_predictions.sum(axis=0).swaplevel().sort_index()
        / diff_predictions.sum(axis=0)
        .groupby("case")
        .transform("sum")
        .swaplevel()
        .sort_index()
    )
    assert np.isclose((market_share.groupby("case").sum() - 1).sum(), 0)

    # remove small contributors
    market_share = market_share.unstack("case")[
        market_share.groupby("subject").mean() > cutoff
    ]
    # rescale
    market_share = market_share.div(market_share.sum())

    return market_share
