import numpy as np 
import func as f

class BP:
    def __init__(self, train_x, train_y, n_iter = 10000, learning_rate = 0.05):
        self.iter = n_iter
        self.lr = learning_rate
        self.x = train_x # no bias
        self.y = train_y
        self.cost = []
        self.setup()

    def setup(self):
        self.set_nn_arch()
        self.set_w()

    def set_nn_arch(self):
        self.input_node = self.x.shape[1]
        self.hidden1 = 10
        self.hidden2 = 5
        self.output_node = self.y.shape[1]
    def set_w(self):
        self.w1 = np.random.random((self.input_node, self.hidden1))
        self.w2 = np.random.random((self.hidden1, self.hidden2))
        self.w3 = np.random.random((self.hidden2, self.output_node))

    def predict(self, x, y):
        self.h1 = f.sigmoid(np.dot(self.x, self.w1))
        self.h2 = f.sigmoid(np.dot(self.h1, self.w2))
        self.res = f.sigmoid(np.dot(self.h2, self.w3))
        self.pred_y = np.where(self.res >= 0.5, 1, 0)
        self.acc = (self.y.shape[0] - abs(self.y - self.pred_y).sum()) / self.y.shape[0] * 100.0

        #return self

    def backp(self):
        E = self.y - self.res
        errors = np.sum(np.square(E)) 
        # Calculate the dirivative
        delta_res = 2 * E * f.derivative_sigmoid(self.res)
        delta_h2 = f.derivative_sigmoid(self.h2) * np.dot(delta_res, self.w3.T)
        delta_h1 = f.derivative_sigmoid(self.h1) * np.dot(delta_h2, self.w2.T)
        # update the coefficient
        # 取 transpose
        self.w3 += self.lr * self.h2.T.dot(delta_res)
        self.w2 += self.lr * self.h1.T.dot(delta_h2)
        self.w1 += self.lr * self.x.T.dot(delta_h1)

        return errors

    def train(self):
        self.error = 0
        for iter in range(0,self.iter):
            self.predict(self.x, self.y)
            self.error = self.backp()
            self.cost.append(self.error)
            if (iter % 1000 == 0):
                print("Accuracy : {}%, loss : {}".format(self.acc, self.error))
            #if (self.acc >= 97.0):
             #   return self
        return self

    def test(self):
        self.predict(self.x, self.y)
        return self


















