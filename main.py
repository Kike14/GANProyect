ticker = "MANU"
start_date = '2014-10-29'
end_date = '2024-10-30'
data = download_data(ticker, start_date, end_date)
precios, data, data_norm, mean, std = preprocess_data(data)
gen_model = generator(data_norm)
disc_model = discriminator()

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
noise = tf.random.normal([100, 2000, 1])

generated_series = gen_model(noise, training=False)

plt.figure(figsize=(12, 6))
for j in range(100):
    plt.plot(generated_series[j, :])

plt.title("Rendientos generadas")
plt.xlabel("Tiempos")
plt.ylabel("Valores de rendimiento")
plt.legend()
plt.show()