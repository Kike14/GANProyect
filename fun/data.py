def download_data(ticker: str, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']]
    return data


def preprocess_data(data: pd.DataFrame):
    prices = data
    r = (np.log(prices[['Adj Close']] / prices[['Adj Close']].shift(1))).dropna()
    mean = r.mean()
    std = r.std()
    r_norm = (r - mean) / std

    return prices, r, r_norm, mean, std