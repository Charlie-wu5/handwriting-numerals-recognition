import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 学习数据
x_train = np.load('x_train.npy')
t_train = np.load('y_train.npy')

# 测试集
x_test = np.load('x_test.npy')

# 数据预处理
x_train, x_test = x_train / 255., x_test / 255.
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
t_train = np.eye(N=10)[t_train.astype("int32").flatten()]

# 数据切分
x_train, x_val, t_train, t_val = \
    train_test_split(x_train, t_train, test_size=10000)


def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e+10))


def create_batch(data, batch_size):
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches:])
    return batched_data


# 随机种子
rng = np.random.RandomState(427)
random_state = 42


def relu(x):
    return np.maximum(0, x)


def deriv_relu(x):
    return (x > 0).astype(np.float64)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=1, keepdims=True)


def deriv_softmax(x):
    s = softmax(x)
    return s * (1 - s)

#交叉熵and l2正则化
def crossentropy_loss(t, y, model=None, lam=0.0):
    ce_loss = -np.mean(np.sum(t * np.log(y + 1e-7), axis=1))
    if model is not None and lam > 0:
        l2 = sum(np.sum(layer.W ** 2) for layer in model.layers)
        return ce_loss + lam * l2
    return ce_loss

# 全连接层
class Dense:
    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.W = np.random.uniform(-0.08, 0.08, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.function = function
        self.deriv_function = deriv_function
        self.x = None
        self.u = None
        self.delta = None

    def __call__(self, x):
        self.x = x
        self.u = np.dot(x, self.W) + self.b
        return self.function(self.u)

    def b_prop(self, delta, W_next):
        self.delta = self.deriv_function(self.u) * np.dot(delta, W_next.T)
        return self.delta

    def compute_grad(self):
        batch_size = self.delta.shape[0]
        self.dW = np.dot(self.x.T, self.delta) / batch_size
        self.db = np.mean(self.delta, axis=0)


class Model:
    def __init__(self, hidden_dims, activation_functions, deriv_functions):
        self.layers = []
        for i in range(len(hidden_dims) - 1):
            self.layers.append(Dense(hidden_dims[i], hidden_dims[i + 1],
                                     activation_functions[i], deriv_functions[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, delta):
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.delta = delta
                layer.compute_grad()
            else:
                delta = layer.b_prop(delta, W)
                layer.compute_grad()
            W = layer.W

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db


lr = 0.1
n_epochs = 20
batch_size = 100

mlp = Model(
    hidden_dims=[784, 256, 128, 10],
    activation_functions=[relu, relu, softmax],
    deriv_functions=[deriv_relu, deriv_relu, deriv_softmax]
)


def train_model(mlp, x_train, t_train, x_val, t_val, n_epochs=10):
    best_val_acc = 0
    patience = 0
    early_stop_patience = 3

    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        x_train, t_train = shuffle(x_train, t_train)
        x_train_batches, t_train_batches = create_batch(x_train, batch_size), create_batch(t_train, batch_size)

        x_val, t_val = shuffle(x_val, t_val)
        x_val_batches, t_val_batches = create_batch(x_val, batch_size), create_batch(t_val, batch_size)

        for x, t in zip(x_train_batches, t_train_batches):
            y = mlp.forward(x)
            loss = crossentropy_loss(t, y, model=mlp, lam=1e-4)
            losses_train.append(loss.tolist())

            delta = y - t
            mlp.backward(delta)
            mlp.update(lr)

            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            train_num += x.shape[0]
            train_true_num += acc

        for x, t in zip(x_val_batches, t_val_batches):
            y = mlp.forward(x)
            loss = crossentropy_loss(t, y)
            losses_valid.append(loss.tolist())

            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            valid_num += x.shape[0]
            valid_true_num += acc

        val_acc = valid_true_num / valid_num

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_true_num / train_num,
            np.mean(losses_valid),
            valid_true_num / valid_num
        ))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"早停触发，在第 {epoch + 1} 轮提前停止训练。最佳验证精度为 {best_val_acc:.4f}")
                break


train_model(mlp, x_train, t_train, x_val, t_val, n_epochs)

# 保存模型参数为 numpy 数组（可以手动重载）
import pickle

with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp, f)

# 示例：生成预测结果
# t_pred = []
# for x in x_test:
#     x = x[np.newaxis, :]
#     y = mlp(x)
#     pred = y.argmax(1).tolist()
#     t_pred.extend(pred)
# submission = pd.Series(t_pred, name='label')
# submission.to_csv('Three_submission_pred.csv', header=True, index_label='id')
