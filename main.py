import numpy as np
import pandas as pd
import tensorflow as tf
import random
from datetime import datetime
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def main():
    # timestamp for the experiment id
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # load data
    train = pd.read_csv("rawdata/train_data.csv")
    test = pd.read_csv("rawdata/test_data.csv")
    traintest = pd.concat([train, test], ignore_index = True, sort = False)

    # list numeric and categorical columns
    num_cols = ["age", "num_child", "service_length", "study_time", "commute", "overtime"]
    cat_cols = ["position", "area", "sex", "partner", "education", "age", "num_child", "service_length"]

    # split data to numeric, categorical and target column(s)
    x_num_trainval = train[num_cols].values
    x_cat_trainval = train[cat_cols].values
    y_trainval = train["salary"].values
    x_num_test = test[num_cols].values
    x_cat_test = test[cat_cols].values

    # tokenize categorical columns
    tokenizer = Tokenizer()
    tokenizer = tokenizer.fit(traintest[cat_cols].values)
    x_cat_trainval = tokenizer.transform(x_cat_trainval)
    x_cat_test = tokenizer.transform(x_cat_test)

    # train
    n = 30
    models = []
    for i in range(n):
        x_num_train, x_num_val, x_cat_train, x_cat_val, y_train, y_val = train_test_split(x_num_trainval, x_cat_trainval, y_trainval, test_size = 0.2, random_state = i)

        model = EmbeddingRegression(
            num_dim = x_num_train.shape[1],
            cat_dim = x_cat_train.shape[1],
            emb_dim = 2,
            vocab_size = tokenizer.vocab_size,
            hidden_units = 128,
            batch_size = 128,
            epochs = 1000,
            patience = 50,
            log_dir = Path("logs", "{}-{:02d}".format(timestamp, i))
        )
        model = model.fit(x_num_train, x_cat_train, y_train, x_num_val, x_cat_val, y_val)
        models.append(model)

    # predict
    y_test = np.zeros(len(test))
    for model in models:
        y_test += model.predict(x_num_test, x_cat_test)
    y_test /= len(models)

    # export predictions
    submit = pd.DataFrame({
        "id": test["id"],
        "y": y_test
    })
    submit.to_csv(Path("submit", "submit_{}.csv".format(timestamp)), header = True, index = False)

class Tokenizer():
    def __init__(self):
        pass

    def fit(self, x):
        self.encoders = []
        self.offsets = []
        self.vocab_size = 0
        for i in range(x.shape[1]):
            encoder = preprocessing.LabelEncoder()
            encoder = encoder.fit(x[:, i])
            self.encoders.append(encoder)
            self.offsets.append(self.vocab_size)
            self.vocab_size += len(encoder.classes_)

        return self

    def transform(self, x):
        x_idx = np.zeros(x.shape)
        for i in range(x.shape[1]):
            x_idx[:, i] = self.encoders[i].transform(x[:, i])
            x_idx[:, i] += self.offsets[i]

        return x_idx

class EmbeddingRegression():
    def __init__(self, num_dim, cat_dim, emb_dim, vocab_size, hidden_units, batch_size, epochs, patience, log_dir):
        # hyper parameters
        self.num_dim = num_dim
        self.cat_dim = cat_dim
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.hidden_units = hidden_units

        # optimizer parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # logging settings
        self.log_dir = log_dir

    def fit(self, x_num_train, x_cat_train, y_train, x_num_val, x_cat_val, y_val):
        # convert np.array to tf.data.Dataset
        ds_train = self.np2ds(x_num_train, x_cat_train, y_train)
        ds_val = self.np2ds(x_num_val, x_cat_val, y_val)

        # build model and compile loss, optimizer and metrics
        self.build()
        self.compile()

        # fit model with early stopping
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(self.epochs):
            for batch_x_num, batch_x_cat, batch_y in ds_train:
                self.train_step(batch_x_num, batch_x_cat, batch_y)
            for batch_x_num, batch_x_cat, batch_y in ds_val:
                self.val_step(batch_x_num, batch_x_cat, batch_y)

            self.print_metrics(epoch)

            if best_loss > self.val_loss.result():
                best_loss = self.val_loss.result()
                best_epoch = epoch
            if epoch - best_epoch > self.patience:
                break

            self.reset_metrics()

        return self

    def predict(self, x_num, x_cat):
        predictions = self.model([x_num, x_cat])
        predictions = predictions.numpy()[:, 0]

        return predictions

    def np2ds(self, x_num, x_cat, y):
        ds = tf.data.Dataset.from_tensor_slices((x_num, x_cat, y))
        ds = ds.shuffle(len(x_num))
        ds = ds.batch(self.batch_size)

        return ds

    def build(self):
        x_num_input = tf.keras.Input(shape = (self.num_dim,), name = "x_num")
        x_cat_input = tf.keras.Input(shape = (self.cat_dim,), name = "x_cat")
        x_cat = tf.keras.layers.Embedding(self.vocab_size, self.emb_dim, name = "embedding")(x_cat_input)
        x_cat = tf.keras.layers.Flatten(name = "flatten")(x_cat)
        x = tf.keras.layers.concatenate([x_num_input, x_cat], name = "concat")
        x = tf.keras.layers.BatchNormalization(name = "hidden_norm1")(x)
        x = tf.keras.layers.Dense(self.hidden_units, activation = tf.keras.layers.ReLU(), name = "hidden_dense1")(x)
        x = tf.keras.layers.BatchNormalization(name = "hidden_norm2")(x)
        x = tf.keras.layers.Dense(self.hidden_units, activation = tf.keras.layers.ReLU(), name = "hidden_dense2")(x)
        x = tf.keras.layers.BatchNormalization(name = "output_norm")(x)
        x = tf.keras.layers.Dense(1, name = "output_dense")(x)
        self.model = tf.keras.Model(inputs = [x_num_input, x_cat_input], outputs = x)

    def compile(self):
        self.loss_object = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name = "train_loss")
        self.train_accuracy = tf.keras.metrics.MeanAbsoluteError(name = "train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name = "val_loss")
        self.val_accuracy = tf.keras.metrics.MeanAbsoluteError(name = "val_accuracy")

    @tf.function
    def train_step(self, x_num, x_cat, y):
        with tf.GradientTape() as tape:
            predictions = self.model([x_num, x_cat])
            loss = self.loss_object(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(y, predictions)

    @tf.function
    def val_step(self, x_num, x_cat, y):
        predictions = self.model([x_num, x_cat])
        loss = self.loss_object(y, predictions)
        self.val_loss(loss)
        self.val_accuracy(y, predictions)

    def print_metrics(self, epoch):
        template = "Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}"
        print(template.format(
            epoch + 1,
            self.train_loss.result(),
            self.train_accuracy.result(),
            self.val_loss.result(),
            self.val_accuracy.result()
        ))

    def reset_metrics(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

if __name__ == "__main__":
    main()
