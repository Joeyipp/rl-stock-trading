import os
import pandas as pd
from functools import reduce

def get_data(path, stock_tickers):
    """
    Return the stock price data (daily open price)
    in TxD sequence where T is # of timesteps (days), D is the feature dimensions (# of stocks)
    """
    stock_tickers = ["timestamp"] + stock_tickers
    stock_dfs = []

    for ticker in stock_tickers[1:]:
        stock_dfs.append(pd.read_csv("{}/stocks/daily_{}.csv".format(path, ticker)))
    
    to_merge = [stock_dfs[0]["timestamp"]]
    for i in range(len(stock_dfs)):
        to_merge.append(stock_dfs[i]["open"])
    
    merged_df = pd.concat(to_merge, join="outer", axis=1)
    merged_df.columns = stock_tickers
    
    # Sort by date (from furthest to recent dates)
    df = merged_df.sort_values(by=['timestamp']).reset_index(drop=True)

    return df


def make_dir(directory):
    """
    Generate directory if not exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    get_data()