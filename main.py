from fun.backtest import backtest
from fun.backtest import backtest2
from fun.data import preprocess_data, download_data
from fun.WGAN import discriminator,generator
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd


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

@tf.function
def train_step(data, batch_size=100):
    noise = tf.random.normal([batch_size, len(data), 1])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = gen_model(noise, training=True)

        y_real = disc_model(data, training=True)
        y_fake = disc_model(generated_data, training=True)

        gen_loss = -tf.math.reduce_mean(
            y_fake)  # o simplemente -tf.math.reduce_mean(y_fake) y sin las funciones de gen_loss y disc_loss
        disc_loss = tf.reduce_mean(y_fake) - tf.reduce_mean(
            y_real)  # o simplemente tf.reduce_mean(y_fake) - tf.reduce_mean(y_real)

    gradients_gen = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_gen, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, disc_model.trainable_variables))

    return gen_loss, disc_loss

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


combinations = [
    (0.01, 0.02),
    (0.01, 0.05),
    (0.01, 0.08),
    (0.02, 0.02),
    (0.02, 0.05),
    (0.02, 0.08),
    (0.03, 0.02),
    (0.03, 0.05),
    (0.03, 0.08),
    (0.025, 0.025)  # Agrega cualquier combinación adicional específica
]

results = []


for sl, tp in combinations:
    calmar_ratios = []

    # Ejecuta 10 simulaciones por combinación
    num_simulations = len(scenarios)
    for i in range(num_simulations):
        # Ejecuta el backtest con la combinación actual
        calmar, _ = backtest(scenarios_df.iloc[:, [i]], sl=sl, tp=tp, n_shares=20)
        calmar_ratios.append(calmar)

    # Calcula la media del Calmar Ratio para esta combinación
    mean_calmar_ratio = np.mean(calmar_ratios)

    # Guarda los resultados
    results.append({
        "sl": sl,
        "tp": tp,
        "mean_calmar_ratio": mean_calmar_ratio
    })

# Imprimir los resultados
for result in results:
    print(f"SL: {result['sl']}, TP: {result['tp']}, Media del Calmar Ratio: {result['mean_calmar_ratio']:.4f}")

trading = backtest2(precios, sl=sl, tp=tp, n_shares=20)
trading