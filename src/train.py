import numpy as np, tensorflow as tf, features as F
CLASSES = ["healthy", "chain_fault", "exhaust_leak"]

def build(n_cls=len(CLASSES)):
    inp = tf.keras.Input(shape=(F.N_FRAMES, F.N_MFCC, 1))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
    return tf.keras.Model(inp, out)

def to_int8_tflite(model, rep_x, path="model_int8.tflite"):
    def rep():
        for i in range(min(300, len(rep_x))):
            yield [rep_x[i:i + 1].astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    open(path, "wb").write(c.convert()); return path

if __name__ == "__main__":     # smoke test with random data
    X = np.random.randn(300, F.N_FRAMES, F.N_MFCC, 1).astype("float32"); y = np.random.randint(0, 3, 300)
    m = build(); m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(X, y, epochs=1, verbose=0); print("params", m.count_params())
    p = to_int8_tflite(m, X); import os; print("tflite bytes", os.path.getsize(p))
    it = tf.lite.Interpreter(model_path=p); it.allocate_tensors(); print(it.get_input_details()[0]["dtype"], it.get_input_details()[0]["quantization"])
