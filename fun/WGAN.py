def generator(data: pd.DataFrame):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(data.shape[0], 1)))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(LSTM(64))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(252))

    return model


def discriminator():
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(252, 1)))  ## (252, 1)
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(LSTM(100))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


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