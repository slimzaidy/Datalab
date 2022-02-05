# 06 Blackbox


## REST API (lokaler service)

- erreichbar unter `http://localhost:8000` (im Folgenden als `{host}` bezeichnet)
- wichtige Hinweise siehe unten

Die untenstehenden Snippets benötigen folgende imports:
```python
import requests
import json
import numpy as np
import tensorflow as tf
```

Aktivierung (GET): `{host}/api/activate?token=$s0m3t0k3n`
```python
host = 'http://127.0.0.1:8000'
token = '123'
response = requests.get(f'{host}/api/activate?token={token}')
print(response.status_code, response.text)
```

Vorhersage (POST): `{host}/evasion/api/predict`
```python

def rand_img():
    return np.random.uniform(0.0, 1.0, size=(32, 32, 3))

img = rand_img.astype(np.float32).reshape(1, 32, 32, 3)
response = requests.post(url, data=img.tobytes(), headers=headers)
pred = np.array(json.loads(response.text)['data'])
```

Challenge erstellen (GET): `{host}/evasion/api/get_challenge`
```python
url = f"{host}/evasion/api/get_challenge"
response = requests.get(url)
response = json.loads(response)
challenge_id, indices, targets = response['id'], response['indices'], response['targets']
```

Challenge lösen (GET): `{host}/evasion/api/solve_challenge`
```python
url = f"{host}/evasion/api/solve_challenge/{challenge_id}"
imgs = np.vstack([rand_img().astype(np.float32).reshape(1, 32, 32, 3) for _ in range(len(indices))])
data = imgs.tobytes()
headers = {'Content-Type': 'application/octet-stream'}
response = requests.post(url, data=data, headers=headers)
```

## Tensorflow Intro
```python
def load_cifar():
    dataset = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = dataset
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cifar()

# create you own model
model = None

# print summary
model.summary()

# gradient calculation
x = tf.constant(x_train[:1])
y = tf.constant(y_train[:1])
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
from tensorflow.keras.utils import to_categorical

# gradient w.r.t. weights
with tf.GradientTape() as tape:
    pred = model(x)
    loss = loss_fn(pred, to_categorical(y, 10))
    tf_grad = tape.gradient(loss, model.trainable_weights)

# gradient w.r.t. inputs
with tf.GradientTape() as tape:
    tape.watch(x)
    pred = model(x)
    loss = loss_fn(pred, to_categorical(y, 10))
    tf_grad = tape.gradient(loss, x)


# manual computation: d/dx f(x) = (f(x_0 + h) - f(x_0 - h)) / (2*h)
def calc_grad(model, x, y, eps=1e-5):
    grad = np.zeros((1, 32, 32, 3), dtype=float)
    y = to_categorical(y, 10)
    for row in range(32):
        for col in range(32):
            for channel in range(3):
                x_pos = x.numpy()
                x_pos[0, row, col, channel] += eps
                x_neg = x.numpy()
                x_neg[0, row, col, channel] -= eps
                x_pos = tf.constant(np.clip(x_pos, 0, 1))
                x_neg = tf.constant(np.clip(x_neg, 0, 1))
                grad[0, row, col, channel] = (loss_fn(model(x_pos), y) - loss_fn(model(x_neg), y)) / (2*eps)
    return grad

grad = calc_grad(model, x, y)

print(np.linalg.norm(np.abs(tf_grad)))
print(np.linalg.norm(np.abs(tf_grad - grad)))
```

## Hinweise

- das Datenformat muss genau stimmen. Wenn nicht, kann es sein, dass der Server keinen Fehler zurückgibt, weil der
  byte stream trotzdem interpretiert werden konnte und das Ergebnis aber keinen Sinn ergibt.
- `evasion/api/predict` funktioniert nur mit einzelnen Bildern (1, 32, 32, 3), nicht mit Batches
- `evasion/api/solve_challenge` nimmt die Daten als Batch (len(indices), 32, 32, 3)
- wichtig: die Pixelwerte müssen zwischen 0 und 1 liegen und vom Typ `np.float32` sein!