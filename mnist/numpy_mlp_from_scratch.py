import numpy as np
import matplotlib.pyplot as plt

class Parameter:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad = np.zeros_like(value)

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Parameter(np.random.randn(in_features, out_features) * 0.01)
        self.b = Parameter(np.zeros(out_features))
    
    def forward(self, x):
        self.input = x
        return x @ self.W.value + self.b.value
    
    def backward(self, grad_output):
        self.W.grad = self.input.T @ grad_output
        self.b.grad = grad_output.sum(axis=0)
        return grad_output @ self.W.value.T

class ReLU:
    def __init__(self):
        self.input = None

    def __call__(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad):
        return grad * (self.input > 0)

class Model:
    def __init__(self):
        self.linear1 = Linear(28*28, 128)
        self.relu1 = ReLU()
        self.linear2 = Linear(128, 10)

    def forward(self, x):
        out = self.linear1.forward(x)
        out = self.relu1(out)
        out = self.linear2.forward(out)
        return out

    def backward(self, grad):
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

    def parameters(self):
        return [self.linear1.W, self.linear1.b, self.linear2.W, self.linear2.b]

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def mse_loss_grad(pred, target):
    return (2 / pred.shape[0]) * (pred - target)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype='>i4')
        return np.frombuffer(f.read(), dtype=np.uint8)

def iterate_minibatches(x, y, batch_size, shuffle=True):
    indices = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, x.shape[0], batch_size):
        end_idx = start_idx + batch_size
        excerpt = indices[start_idx:end_idx]
        yield x[excerpt], y[excerpt]

if __name__ == "__main__":
    learning_rate = 1e-3
    epoch = 30
    batch_size = 128

    model = Model()

    # Load from correct file paths
    x_train = load_images('mnist/train-images.idx3-ubyte')
    y_train = load_labels('mnist/train-labels.idx1-ubyte')
    x_test = load_images('mnist/t10k-images.idx3-ubyte')
    y_test = load_labels('mnist/t10k-labels.idx1-ubyte')
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    print("Train:", x_train.shape, y_train.shape)
    print("Test:", x_test.shape, y_test.shape)

    for e in range(epoch):
        total_loss = 0
        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size):
            output = model.forward(x_batch)
            loss = mse_loss(output, y_batch)
            loss_grad = mse_loss_grad(output, y_batch)

            model.backward(loss_grad)

            for param in model.parameters():
                param.value -= learning_rate * param.grad

            total_loss += loss * x_batch.shape[0]

        avg_loss = total_loss / x_train.shape[0]
        print(f"Epoch {e} | Loss: {avg_loss}")
    
    output = model.forward(x_test)
    pred_classes = np.argmax(output, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    index = 10
    output = model.forward(x_test[index].reshape(1, -1))
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title("Label: " + str(np.argmax(y_test[index])) + ", Prediction: " + str(np.argmax(output[0])))
    plt.axis('off')
    plt.show()
