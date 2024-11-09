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


def backtest2(data: pd.DataFrame, sl: float, tp: float, n_shares: int, rf=0):
    data = data.copy()

    # Asegurar que las columnas OHLC existen
    if 'Open' not in data.columns:
        data['Open'] = data['Close']
    if 'High' not in data.columns:
        data['High'] = data['Close']
    if 'Low' not in data.columns:
        data['Low'] = data['Close']

    # Calcular las Bandas de Bollinger y señales de compra
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20)
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Buy'] = bollinger.bollinger_lband_indicator()

    capital = 1_000_000
    COM = 0.125 / 100  # Comisión en porcentaje
    active_long_positions = []
    portfolio_value = [capital]
    cash_through_time = [capital]
    operations = []  # Lista para almacenar cada operación
    buy_signals = pd.Series(np.nan, index=data.index)  # Para almacenar precios de compra con NaN por defecto

    wins = 0
    losses = 0

    # Iterar sobre los datos del mercado
    for i, row in data.iterrows():
        long_signal = row['BB_Buy']  # Señal de compra

        # Entrada de posición larga
        if long_signal == True:
            cost = row['Close'] * n_shares * (1 + COM)
            if capital > cost and len(active_long_positions) < 100:
                capital -= cost
                position = Position(ticker="MANU", price=row['Close'], n_shares=n_shares)
                active_long_positions.append(position)
                operations.append({
                    'Type': 'Buy',
                    'Price': row['Close'],
                    'Shares': n_shares,
                    'Total Cost': cost,
                    'Date': row.name
                })
                buy_signals[row.name] = row['Close']  # Almacenar el precio de compra en la fecha correspondiente

        # Cierre de posiciones largas
        for position in active_long_positions.copy():
            if row['Close'] > position.price * (1 + tp):
                revenue = row['Close'] * position.n_shares * (1 - COM)
                capital += revenue
                wins += 1
                active_long_positions.remove(position)
                operations.append({
                    'Type': 'Sell (Take Profit)',
                    'Price': row['Close'],
                    'Shares': position.n_shares,
                    'Total Revenue': revenue,
                    'Date': row.name
                })
            elif row['Close'] < position.price * (1 - sl):
                revenue = row['Close'] * position.n_shares * (1 - COM)
                capital += revenue
                losses += 1
                active_long_positions.remove(position)
                operations.append({
                    'Type': 'Sell (Stop Loss)',
                    'Price': row['Close'],
                    'Shares': position.n_shares,
                    'Total Revenue': revenue,
                    'Date': row.name
                })

        # Valor del portafolio y efectivo en el tiempo
        value = capital + sum(p.n_shares * row['Close'] for p in active_long_positions)
        portfolio_value.append(value)
        cash_through_time.append(capital)

    # Convertir portfolio_value y cash_through_time a Series de pandas
    portfolio_series = pd.Series(portfolio_value)
    cash_series = pd.Series(cash_through_time)

    # Calcular el rendimiento logarítmico
    portafolio_value_rends = np.log(portfolio_series / portfolio_series.shift(1))

    # Calcular el Sharpe Ratio
    mean_portfolio_return = portafolio_value_rends.mean()
    portfolio_volatility = portafolio_value_rends.std()
    sharpe_ratio = (mean_portfolio_return - rf) / portfolio_volatility

    # Calcular el rendimiento del portafolio a lo largo del periodo
    initial_value = portfolio_series.iloc[0]
    final_value = portfolio_series.iloc[-1]
    portfolio_return = ((final_value - initial_value) / initial_value) * 100  # Porcentaje

    # Calmar Ratio
    calmar_value = calmar_ratio(portafolio_value_rends)

    # Gráfico de Velas con Banda Baja de Bollinger y Señales de Compra
    add_plot = [
        mpf.make_addplot(data['BB_Low'], color='green', label='Banda Baja de Bollinger'),  # Solo muestra la banda baja
        mpf.make_addplot(buy_signals, scatter=True, markersize=50, marker='o', color='purple', label='Señal de Compra')
        # Señales de compra
    ]

    # Crear un gráfico de línea
    fig, axlist = mpf.plot(data, type='line', style='charles', addplot=add_plot,
                           title="Bandas de Bollinger y Señales de Compra", returnfig=True)

    # Añadir una leyenda manualmente
    axlist[0].legend(['Precio de Cierre', 'Banda Baja de Bollinger', 'Señal de Compra'], loc='upper left')

    # Mostrar el gráfico del valor del portafolio y efectivo en el tiempo
    plt.figure(figsize=(14, 7))
    plt.plot(cash_series, label='Efectivo en el Tiempo', color='orange')
    plt.title("Efectivo en el Tiempo")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))

    # Gráfico del valor del portafolio y el precio a lo largo del tiempo (usando dos ejes)
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Gráfico del precio en el eje primario (eje izquierdo)
    ax1.plot(data['Close'].values, label='Precio de Cierre', color='blue')
    ax1.set_xlabel("Índice")
    ax1.set_ylabel("Precio", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Gráfico del valor del portafolio en el eje secundario (eje derecho)
    ax2 = ax1.twinx()
    ax2.plot(portfolio_series.values, label='Valor del Portafolio', color='green')
    ax2.set_ylabel("Valor del Portafolio", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Título y leyenda
    plt.title("Precio de Cierre y Valor del Portafolio a través del Tiempo")
    fig.tight_layout()  # Ajusta el diseño para evitar solapamiento
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

    # Retornar resultados y métricas
    return {
        'Calmar Ratio': calmar_value,
        'Sharpe Ratio': sharpe_ratio,
        'Win-Loss Ratio': wins / losses if losses > 0 else np.inf,
        'Portfolio Return (%)': portfolio_return
    }

def calmar_ratio(returns):
    max_drawdown = (returns.cummax() - returns).max()
    annual_return = returns.mean() * 252  # Asumiendo 252 días de trading en un año
    return annual_return / abs(max_drawdown)
