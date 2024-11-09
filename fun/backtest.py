class Position:
    def _init_(self, ticker: str, price: float, n_shares: int):
        self.ticker = ticker
        self.price = price
        self.n_shares=n_shares

def backtest(data: pd.DataFrame, sl: float, tp: float,
             n_shares: int, rf=0):
    data = data.copy()

    data.columns.values[0] = "Close"

    bollinger = ta.volatility.BollingerBands(data.Close, window=20)
    data['BB_Buy'] = bollinger.bollinger_lband_indicator()

    capital = 1_000_000
    COM = 0.125 / 100  # Commission percentage
    active_long_positions = []
    portfolio_value = [capital]

    wins = 0
    losses = 0

    # Iterar sobre los datos del mercado
    for i, row in data.iterrows():
        long_signal = row.BB_Buy  # Señal de compra

        # Entrada de posición larga
        if long_signal == True:
            cost = row.Close * n_shares * (1 + COM)
            if capital > cost and len(active_long_positions) < 100:
                capital -= row.Close * n_shares * (1 + COM)
                active_long_positions.append(
                    Position(ticker="MANU", price=row.Close, n_shares=n_shares))

        # Cierre de posiciones largas
        for position in active_long_positions.copy():
            if row.Close > position.price * (1 + tp):
                capital += row.Close * position.n_shares * (1 - COM)
                wins += 1  # Operación ganadora
                active_long_positions.remove(position)
            elif row.Close < position.price * (1 - sl):
                capital += row.Close * position.n_shares * (1 - COM)
                losses += 1  # Operación perdedora
                active_long_positions.remove(position)

        value = capital + len(active_long_positions) * n_shares * row.Close
        portfolio_value.append(value)

    # Convertir portfolio_value a una Serie de pandas
    portfolio_series = pd.Series(portfolio_value)

    # Calcular el rendimiento logarítmico
    portafolio_value_rends = np.log(portfolio_series / portfolio_series.shift(1))

    # Calcular el Sharpe Ratio
    mean_portfolio_return = portafolio_value_rends.mean()  # Rendimiento promedio del portafolio
    portfolio_volatility = portafolio_value_rends.std()  # Volatilidad del portafolio
    sharpe_ratio = (mean_portfolio_return - rf) / portfolio_volatility  # Sharpe Ratio

    # print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Calcular el valor máximo acumulado en cada momento
    running_max = portfolio_series.cummax()

    # Calcular el Drawdown
    drawdown = (portfolio_series - running_max) / running_max

    # Max Drawdown
    max_drawdown = drawdown.min()

    # print(f"Max Drawdown: {max_drawdown:.4f}")

    # Calcular el Win-Loss Ratio
    if losses > 0:
        win_loss_ratio = wins / losses
    else:
        win_loss_ratio = np.inf  # Si no hay pérdidas, el Win-Loss ratio es infinito

    passive = list(data.Close)

    # print(f"Win-Loss Ratio: {win_loss_ratio:.2f}")

    calmar_value = calmar_ratio(portafolio_value_rends)

    return calmar_value, portfolio_series

def calmar_ratio(returns):
    max_drawdown = (returns.cummax() - returns).max()
    annual_return = returns.mean() * 252  # Asumiendo 252 días de trading en un año
    return annual_return / abs(max_drawdown)