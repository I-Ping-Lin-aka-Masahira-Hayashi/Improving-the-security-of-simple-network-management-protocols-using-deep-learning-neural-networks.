#Below is my sample
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# SNMP-MIB データの生成（実際のデータセットに置き換える必要があります）
def generate_sample_data(n_samples=1000, n_features=10):
    return np.random.randn(n_samples, n_features)

# データの前処理
def preprocess_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# ジェネレーターモデルの定義
def build_generator(latent_dim, output_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_dim, activation='tanh')
    ])
    return model

# ディスクリミネーターモデルの定義
def build_discriminator(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# GANモデルの定義
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
        
        # ジェネレーターの訓練
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        generated_data = self.generator(random_latent_vectors)
        combined_data = tf.concat([generated_data, real_data], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # ジェネレーターの訓練
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# 異常検知関数
def detect_anomalies(gan, data, threshold):
    latent_dim = gan.generator.input_shape[-1]
    random_latent_vectors = tf.random.normal(shape=(len(data), latent_dim))
    generated_data = gan.generator(random_latent_vectors)
    
    real_scores = gan.discriminator(data)
    fake_scores = gan.discriminator(generated_data)
    
    anomaly_scores = np.abs(real_scores - fake_scores)
    anomalies = anomaly_scores > threshold
    return anomalies, anomaly_scores

# メイン関数
def main():
    # データの生成と前処理
    raw_data = generate_sample_data()
    data = preprocess_data(raw_data)
    train_data, test_data = train_test_split(data, test_size=0.2)

    # モデルの構築
    latent_dim = 100
    input_dim = train_data.shape[1]
    generator = build_generator(latent_dim, input_dim)
    discriminator = build_discriminator(input_dim)
    gan = GAN(generator, discriminator)

    # モデルのコンパイルと訓練
    gan.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss_fn=keras.losses.BinaryCrossentropy()
    )
    
    gan.fit(train_data, epochs=100, batch_size=32)

    # 異常検知の実行
    threshold = 0.5  # この値は実験的に調整したほうがいい
    anomalies, anomaly_scores = detect_anomalies(gan, test_data, threshold)

    print(f"Anomalies detected: {np.sum(anomalies)}")
    print(f"Average of anomaly scores: {np.mean(anomaly_scores)}")

if __name__ == "__main__":
    main()