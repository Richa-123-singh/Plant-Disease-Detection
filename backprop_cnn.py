from builtins import object
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import gc


def forward_convolution(x, w, b, conv_param):
    stride = conv_param['stride']
    pad = conv_param['pad']
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out))

    for i in range(N):
        for j in range(F):
            for k in range(H_out):
                for l in range(W_out):
                    h_start, w_start = k * stride, l * stride
                    h_end, w_end = h_start + HH, w_start + WW
                    x_slice = x_padded[i, :, h_start:h_end, w_start:w_end]
                    out[i, j, k, l] = np.sum(x_slice * w[j]) + b[j]

    cache = (x, w, b, conv_param)
    return out, cache

def backward_convolution(dout, cache):
	dx, dw, db = None, None, None
	x, w, b, conv_param = cache
	stride = conv_param['stride']
	pad = conv_param['pad']
	N, F, Hout, Wout = dout.shape
	dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
	hfilt, wfilt = w.shape[2], w.shape[3]
	xpadded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))
	dxpadded = np.zeros(xpadded.shape)
	for n in range(N):
		for f in range(F):
			for hi in range(Hout):
				for wi in range(Wout):
					dxpadded[n, :, hi*stride:hi*stride+hfilt, wi*stride:wi*stride+wfilt] += w[f, :, :, :]*dout[n, f, hi, wi]
					dw[f, :, :, :] += xpadded[n, :, hi*stride:hi*stride+hfilt, wi*stride:wi*stride+wfilt]*dout[n, f, hi, wi]
					db[f] += dout[n, f, hi, wi]
	return dx, dw, db

def forward_maxpool(x, pool_param):
    N, C, H, W = x.shape
    PW, PH, stride = pool_param['pool_width'], pool_param['pool_height'], pool_param['stride']
    
    h_out = 1 + (H - PH) // stride
    w_out = 1 + (W - PW) // stride
    
    out = np.zeros((N, C, h_out, w_out))
    
    for h_idx in range(0, H - PH + 1, stride):
        for w_idx in range(0, W - PW + 1, stride):
            pool_region = x[:, :, h_idx:h_idx+PH, w_idx:w_idx+PW]
            out[:, :, h_idx//stride, w_idx//stride] = np.max(pool_region, axis=(2, 3))
    
    cache = (x, pool_param)
    return out, cache

def backward_maxpool(dout, cache):
    dx = None
    x, pool_param = cache
    PW, PH, stride = pool_param['pool_width'], pool_param['pool_height'], pool_param['stride']
    N, C, Hout, Wout = dout.shape
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for h in range(Hout):
                for w in range(Wout):
                    x_region = x[n, c, h * stride:h * stride + PH, w * stride:w * stride + PW]
                    max_index = np.unravel_index(x_region.argmax(), x_region.shape)
                    
                    diff_matrix = np.zeros_like(x_region)
                    diff_matrix[max_index] = 1
                    
                    dx[n, c, h * stride:h * stride + PH, w * stride:w * stride + PW] += diff_matrix * dout[n, c, h, w]
    return dx

def softmax_loss_with_gradient(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    exp_shifted_logits = np.exp(shifted_logits)
    sum_exp_shifted_logits = np.sum(exp_shifted_logits, axis=1, keepdims=True)
    probs = exp_shifted_logits / sum_exp_shifted_logits
    
    log_probs = np.log(probs)
    
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def relu_custom_forward(x):
    out = np.zeros_like(x)
    out[x > 0] = x[x > 0]
    cache = x
    
    return out, cache

def relu_custom_backward(dout, cache):
    dx, x = None, cache
    mask = (x > 0).astype(float)
    dx = dout * mask
    return dx

def linear_forward(x, w, b):
	out = np.dot(x.reshape((x.shape[0], w.shape[0])), w) + b
	cache = (x, w, b)
	return out, cache

def linear_backward(dout, cache):
	x, w, b = cache
	dx, dw, db = None, None, None
	dx = np.dot(dout, w.T).reshape(x.shape)
	dw = np.dot(x.reshape((x.shape[0], w.shape[0])).T, dout)
	db = np.dot(np.ones((1, x.shape[0])), dout)
	return dx, dw, db


def forward_relupool(x, w, b, conv_param, pool_param):
    a, conv_cache = forward_convolution(x, w, b, conv_param)
    s, relu_cache = relu_custom_forward(a)
    out, pool_cache = forward_maxpool(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def backward_relupool(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    ds = backward_maxpool(dout, pool_cache)
    da = relu_custom_backward(ds, relu_cache)
    dx, dw, db = backward_convolution(da, conv_cache)
    return dx, dw, db


def custom_forward_relu(x, w, b):
    a, fc_cache = linear_forward(x, w, b)
    out, relu_cache = relu_custom_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def custom_backward_relu(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_custom_backward(dout, relu_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db


class ThreeLayerConvNet(object):
	def __init__(self, input_dim=(3, 256, 256), num_filters=32, filter_size=3,
					hidden_dim=100, num_classes=16, weight_scale=1e-3, reg=0.0,
					dtype=np.float32):
		self.params = {}
		self.reg = reg
		self.dtype = dtype

		self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size = (num_filters, input_dim[0], filter_size, filter_size))
		self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size = (np.prod((num_filters,input_dim[1],input_dim[2]))//4,hidden_dim))
		self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size = (hidden_dim, num_classes))
		self.params['b1'] = np.zeros(num_filters)
		self.params['b2'] = np.zeros(hidden_dim)
		self.params['b3'] = np.zeros(num_classes)

		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)


	def loss(self, X, y=None):
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		scores = None
		crp_out, crp_cache = forward_relupool(X, W1, b1, conv_param, pool_param)
		ar_out, ar_cache = custom_forward_relu(crp_out, W2, b2)
		a_out, a_cache = linear_forward(ar_out, W3, b3)
		scores = a_out
		
		if y is None:
			return scores

		loss, grads = 0, {}
		softmax, softmax_grad = softmax_loss_with_gradient(scores, y)
		loss = softmax + 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
		a_grad, grads['W3'], grads['b3'] = linear_backward(softmax_grad, a_cache)
		ar_grad, grads['W2'], grads['b2'] = custom_backward_relu(a_grad, ar_cache)
		_, grads['W1'], grads['b1'] = backward_relupool(ar_grad, crp_cache)
		
		# add regularization gradient
		grads['W3'] += self.reg*(W3)
		grads['W2'] += self.reg*(W2)
		grads['W1'] += self.reg*(W1)

		return loss, grads


print("Here 0")
data = np.load("./data_10.npy")
label = np.load("./label_10.npy")
data, label = shuffle(data, label)
data = np.array(data).reshape(-1,3,256,256)
label = np.array(label)
split = int(data.shape[0]*0.8)
print("Here 1")
X_train = data[:split]
y_train = label[:split]
X_val = data[split:]
y_val = label[split:]
print("Here 2")


model = ThreeLayerConvNet(weight_scale=1e-2)
def sgd(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw.reshape(w.shape)
    return w, config

update_rule = 'sgd'
optim_config = {}
optim_configs = {}
lr_decay = 1.0
checkpoint_name = "Test_updated_"
batch_size = 4
epochs = 5
num_train_samples = 1000
best_params = {}
loss_history = []
train_acc_history = []
val_acc_history = []
num_val_samples = None


epoch = 0


def step():
        num_train = X_train.shape[0]
        batch_mask = np.random.choice(num_train, batch_size)
        X_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]

        loss, grads = model.loss(X_batch, y_batch)
        loss_history.append(loss)

        for p, w in model.params.items():
            dw = grads[p]
            optim_configs = {}
            next_w, next_config = sgd(w, dw, {'learning_rate': 1e-2})
            model.params[p] = next_w
            optim_configs[p] = next_config
def accuracy( X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = model.loss(X=X[start:end])
            y_pred.append(np.argmax(scores,axis=1)) 
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

def checkpoint():
        if checkpoint_name is None: return
        checkpoint = {
          'model': model,
          'update_rule': update_rule,
          'lr_decay': lr_decay,
          'optim_config': optim_config,
          'batch_size': batch_size,
          'num_train_samples': num_train_samples,
          'num_val_samples': num_val_samples,
          'epoch': epoch,
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (checkpoint_name, epoch)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

best_val_acc = 0

def train():
    best_val_acc = 0
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    num_iterations = epochs * iterations_per_epoch
    epoch = 0
    for t in range(num_iterations):
        gc.collect()
        step()
        gc.collect()
        print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, loss_history[-1]))

        epoch_end = (t + 1) % iterations_per_epoch == 0
        if epoch_end:
            epoch += 1
            for k in optim_configs:
                optim_configs[k]['learning_rate'] *= lr_decay

        first_it = (t == 0)
        last_it = (t == num_iterations - 1)
        if first_it or last_it or epoch_end:
            train_acc = accuracy(X_train, y_train,
                    num_samples=num_train_samples)
            val_acc = accuracy(X_val, y_val,
                    num_samples=num_val_samples)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            checkpoint()
            gc.collect()
            print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        epoch, epochs, train_acc, val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {}
                for k, v in model.params.items():
                    best_params[k] = v.copy()

    # At the end of training swap the best params into the model
    model.params = best_params
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, '-o')
    plt.plot(val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


y_train = label[:split]
y_val = label[split:]
train()



