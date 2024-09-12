#my sample
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Generate synthetic SNMP-like data
def generate_synthetic_snmp_data(n_samples=10000, n_features=5):
    data = np.zeros((n_samples, n_features))
    
    # SNMP version (0: v1, 1: v2c, 2: v3)
    data[:, 0] = np.random.choice([0, 1, 2], n_samples)
    
    # Community string length (typically 6-15 characters)
    data[:, 1] = np.random.randint(6, 16, n_samples)
    
    # PDU type (0: get, 1: get-next, 2: set, 3: trap, etc.)
    data[:, 2] = np.random.randint(0, 5, n_samples)
    
    # Number of variable bindings (typically 1-10)
    data[:, 3] = np.random.randint(1, 11, n_samples)
    
    # Packet length (typically 60-1500 bytes)
    data[:, 4] = np.random.randint(60, 1501, n_samples)
    
    return data

# Data preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Generator model
def build_generator(latent_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(output_dim, activation='tanh')
    ])
    return model

# Discriminator model
def build_discriminator(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
class GAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        latent_dim = self.generator.input_shape[-1]
        
        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        generated_data = self.generator(random_latent_vectors)
        combined_data = tf.concat([generated_data, real_data], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# Anomaly detection function
def detect_anomalies(gan, data, threshold):
    latent_dim = gan.generator.input_shape[-1]
    random_latent_vectors = tf.random.normal(shape=(len(data), latent_dim))
    generated_data = gan.generator(random_latent_vectors)
    
    real_scores = gan.discriminator(data)
    fake_scores = gan.discriminator(generated_data)
    
    anomaly_scores = np.abs(real_scores - fake_scores)
    anomalies = anomaly_scores > threshold
    return anomalies, anomaly_scores

# Main function
def main():
    # Generate and preprocess synthetic SNMP data
    raw_data = generate_synthetic_snmp_data()
    data = preprocess_data(raw_data)
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Model building and training
    latent_dim = 100
    input_dim = train_data.shape[1]
    generator = build_generator(latent_dim, input_dim)
    discriminator = build_discriminator(input_dim)
    gan = GAN(generator, discriminator)

    gan.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss_fn=keras.losses.BinaryCrossentropy()
    )
    
    gan.fit(train_data, epochs=100, batch_size=32)

    # Anomaly detection
    threshold = 0.5  # Adjust this based on my specific requirements
    anomalies, anomaly_scores = detect_anomalies(gan, test_data, threshold)

    print(f"Anomalies detected: {np.sum(anomalies)}")
    print(f"Average anomaly score: {np.mean(anomaly_scores)}")

    # Additional step: Alert on detected anomalies
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            print(f"Potential security threat detected in synthetic packet {i}")

if __name__ == "__main__":
    main()