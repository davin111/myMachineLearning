"""
Minimal character-level LSTM model. Written by Byeon Davin (Original ver. @karpathy)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import time



MaxIter = 5000
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size*4, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size*4, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size*4, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lossFun(inputs, targets, hprev, cprev):
    xs, hs, cs, is_, fs, os, gs, ys, ps= {}, {}, {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)
    
    loss = 0
    H = hidden_size
    
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        tmp = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh  # hidden state
        is_[t] = sigmoid(tmp[:H])
        fs[t] = sigmoid(tmp[H:H*2])
        os[t] = sigmoid(tmp[H*2:H*3])
        gs[t] = np.tanh(tmp[H*3:])
        
        cs[t] = fs[t] * cs[t-1] + is_[t] * gs[t]
        hs[t] = os[t] * np.tanh(cs[t])
    
        # compute loss
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
    
    
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])
    n = 1
    a = len(targets) - 1
    
    for t in reversed(range(len(inputs))):
        if n > len(targets):
            continue
        dy = np.copy(ps[t])
        dy[targets[a]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dc = dcnext + (1 - np.tanh(cs[t]) * np.tanh(cs[t])) * dh * os[t]  # backprop through tanh nonlinearity
        dcnext = dc * fs[t]
        di = dc * gs[t]
        df = dc * cs[t-1]
        do = dh * np.tanh(cs[t])
        dg = dc * is_[t]
        ddi = (1 - is_[t]) * is_[t] * di # sigmoid_grad
        ddf = (1 - fs[t]) * fs[t] * df
        ddo = (1 - os[t]) * os[t] * do
        ddg = (1 - gs[t] * gs[t]) * dg
        da = np.hstack((ddi.ravel(),ddf.ravel(),ddo.ravel(),ddg.ravel()))
        dWxh += np.dot(da[:,np.newaxis],xs[t].T)
        dWhh += np.dot(da[:,np.newaxis],hs[t-1].T)
        dbh += da[:, np.newaxis]
        dhnext = np.dot(Whh.T, da[:, np.newaxis])
        n += 1
        a -= 1
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]


def sample(h, c, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is hidden state, c is cell state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
  
    H = hidden_size

    for t in xrange(n):
        tmp = np.dot(Wxh, x) + np.dot(Whh, h) + bh  # hidden state
        is_ = sigmoid(tmp[:H])
        fs = sigmoid(tmp[H:H*2])
        os = sigmoid(tmp[H*2:H*3])
        gs = np.tanh(tmp[H*3:])
        
        c = fs * c + is_ * gs
        h = os * np.tanh(c)

        y = np.dot(Why, h) + by  # unnormalized log probabilities for next chars
        p = np.exp(y) / np.sum(np.exp(y))  # probabilities for next chars

        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    return ixes



n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

loss_list = []
iter_list = []
start_time = time.time()
tmp_time = 0
sum_time = 0


while n <= MaxIter:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # reset LSTM memory
        cprev = np.zeros((hidden_size,1))
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]


    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, cprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print '-------\n %s \n-------' % (txt, )

  
    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
  
  
    if n % 100 == 0:
        if n != 0:
            tmp_time = time.time() - start_time
            sum_time += tmp_time
            print 'iter %d, loss: %f || %s sec/iter 100' % (n, smooth_loss, tmp_time) # print progress
        else:
            print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
        iter_list.append(n)
        loss_list.append(smooth_loss)
        start_time = time.time()

        if n == MaxIter:
            print '%s sec/iter %d' % (sum_time, MaxIter)


  # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter


# draw loss and iter
plt.title("LSTM || x: iter / y: loss")
plt.plot(iter_list, loss_list)

plt.show()
