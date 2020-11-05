import os
import numpy as np
import pandas
import datetime
from pandas import DataFrame, Series
import tushare as ts
from . import simple_logging as logging

log = logging.getLogger()
data_dirname = 'data'
data_filename_template = '{code}-daily.csv'
base_date = datetime.date(1970, 1, 1)

tushare_access_token = 'fc6e27a7c03efce2b7a53b2f39eba8d1b9daba7c35da6c01b16f8be7'
api = ts.pro_api(tushare_access_token)
specific_api = api.index_daily


def _parse_frame(df: DataFrame, series: str):
    dates: np.ndarray = df.trade_date.to_numpy()
    dates = np.fromiter(map(lambda d: (datetime.datetime.strptime(str(d), '%Y%m%d').date() - base_date).days, dates),
                        dtype=int)
    prices: np.ndarray = getattr(df, series).to_numpy()
    dates = dates[:, None]
    prices = prices[:, None]
    return dates, prices


def _fetch_offline_data(code: str, series: str):
    path = os.path.join(data_dirname, data_filename_template.format(code=code))
    if os.path.exists(path):
        df: DataFrame = pandas.read_csv(path)
        return _parse_frame(df, series)


def _fetch_online_data(code: str, series: str):
    path = os.path.join(data_dirname, data_filename_template.format(code=code))
    os.makedirs(data_dirname, exist_ok=True)
    df: DataFrame = specific_api(ts_code=code)
    df.to_csv(path)
    return _parse_frame(df, series)


def fetch_data(code: str, series: str):
    res = _fetch_offline_data(code, series)
    if res is not None:
        log.info(f'Loaded offline data for {code} - {series}')
    else:
        res = _fetch_online_data(code, series)
        log.info(f'Loaded online data for {code} - {series}')
    return res
