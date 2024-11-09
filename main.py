ticker = "MANU"
start_date = '2014-10-29'
end_date = '2024-10-30'
data = download_data(ticker, start_date, end_date)
precios, data, data_norm, mean, std = preprocess_data(data)
gen_model = generator(data_norm)
disc_model =discriminator()
precios = precios.rename(columns={'Adj Close': 'Close'})

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
gen_loss_history = []
disc_loss_history = []

num_batches = data_norm.shape[0] // 200
for epoch in range(100):
    for i in tqdm.tqdm(range(num_batches)):
        batch = data_norm[i*200:(1+i)*200]
        gen_loss, disc_loss = train_step(batch)

        gen_loss_history.append(gen_loss.numpy())
        disc_loss_history.append(disc_loss.numpy())


plt.plot(gen_loss_history)  # se tiene que acercar a cero. por que?
plt.plot(disc_loss_history) # se tiene que alejar mas. por que?

gen2 = tf.keras.models.load_model("./generador.keras")
noise = tf.random.normal([100, 2000, 1])

generated_series = gen2(noise, training=False)


plt.figure(figsize=(12, 6))
for j in range(100):
    plt.plot(generated_series[j, :])

plt.title("Rendientos generadas")
plt.xlabel("Tiempos")
plt.ylabel("Valores de rendimiento")
plt.legend()
plt.show()

scenarios = generated_series.numpy().tolist()

data_n = []

for scenario in scenarios:
    S0 = precios['Adj Close'].sample(n=1).iloc[0]
    prices = [S0]
    for log_return in scenario:
        next_price = prices[-1] * np.exp(log_return)
        prices.append(next_price)
    data_n.append(prices)

for prices in data_n:
    plt.plot(prices, alpha=0.5, linewidth=0.75)

plt.plot((precios.iloc[:253]).values, label='Real Price', color='black', linewidth=1.5)
plt.title('Simulated Prices vs Real Price')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

scenarios_df = pd.DataFrame()
for i in range(len(data_n)):
    scenarios_df[f'Simulación {i + 1}'] = data_n[i]

scenarios_df['Close'] = precios['Adj Close'].iloc[:253].values  # agregar precio original
scenarios_df
