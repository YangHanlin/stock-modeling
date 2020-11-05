import pandas
import tushare as ts
import os
from datetime import date


def old_style():
    # dirname = f'data-{date.today().isoformat()}'
    dirname = 'data'
    os.makedirs(dirname, exist_ok=True)
    codes = [
        'sh', 'sz', 'hs300', 'sz50', 'zxb', 'cyb'
    ]
    ktypes = [
        'D', 'W', 'M', '5', '15', '30', '60'
    ]
    for code in codes:
        for ktype in ktypes:
            df: pandas.DataFrame = ts.get_hist_data(code=code, ktype=ktype)
            print(f'Getting {ktype} data for {code}: {df.size} fetched')
            df.to_csv(f'{dirname}/{code}-{ktype}.csv')


def new_style():
    # dirname = f'data-new-{date.today().isoformat()}'
    dirname = 'data'
    os.makedirs(dirname, exist_ok=True)
    code = '600519.SH'
    api = ts.pro_api()
    df: pandas.DataFrame = api.daily(ts_code=code)
    print(f'Getting data for {code}: {df.count(0)} fetched')
    df.to_csv(f'{dirname}/{code}-daily.csv')


def main():
    new_style()


if __name__ == '__main__':
    main()
