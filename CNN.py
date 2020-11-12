import warnings
import os
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN:
    @classmethod
    def initializeFilter(cls, size, scale=1.0):
        stddev = scale / np.sqrt(np.prod(size))
        return np.random.normal(loc=0, scale=stddev, size=size)

    @classmethod
    def initializeWeight(cls, size):
        return np.random.standard_normal(size=size) * 0.01

    @classmethod
    def convolutionBackward(cls, dconv_prev, conv_in, filt, s):
        (n_f, n_c, f, _) = filt.shape
        (_, orig_dim, _) = conv_in.shape
        ## initialize derivatives
        dout = np.zeros(conv_in.shape)
        dfilt = np.zeros(filt.shape)
        dbias = np.zeros((n_f, 1))
        for curr_f in range(n_f):
            # loop through all filters
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # loss gradient of filter (used to update the filter)
                    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
                    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                    dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            # loss gradient of the bias
            dbias[curr_f] = np.sum(dconv_prev[curr_f])

        return dout, dfilt, dbias

    @classmethod
    def nanargmax(cls, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs

    @classmethod
    def maxpoolBackward(cls, dpool, orig, f, s):
        (n_c, orig_dim, _) = orig.shape

        dout = np.zeros(orig.shape)

        for curr_c in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # obtain index of largest value in input for current window
                    (a, b) = cls.nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                    dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]

                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1

        return dout

    @classmethod
    def maxpool(cls, image, f=2, s=2):
        n_c, h_prev, w_prev = image.shape

        # calculate output dimensions after the maxpooling operation.
        h = int((h_prev - f) / s) + 1
        w = int((w_prev - f) / s) + 1

        # create a matrix to hold the values of the maxpooling operation.
        downsampled = np.zeros((n_c, h, w))

        # slide the window over every part of the image using stride s. Take the maximum value at each step.
        for i in range(n_c):
            curr_y = out_y = 0
            # slide the max pooling window vertically across the image
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                # slide the max pooling window horizontally across the image
                while curr_x + f <= w_prev:
                    # choose the maximum value within the window at each step and store it to the output matrix
                    downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        return downsampled

    @classmethod
    def convolution(cls, image, filt, bias, s=1):
        (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
        n_c, in_dim, _ = image.shape  # image dimensions

        out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions

        assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

        out = np.zeros((n_f, out_dim, out_dim))

        # convolve the filter over every part of the image, adding the bias at each step.
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                                bias[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1

        return out

    @classmethod
    def conv(cls, image, label, params, conv_s, pool_f, pool_s):

        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        conv1 = cls.convolution(image, f1, b1, conv_s)  # convolution operation
        conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

        conv2 = cls.convolution(conv1, f2, b2, conv_s)  # second convolution operation
        conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

        pooled = cls.maxpool(conv2, pool_f, pool_s)  # maxpooling operation

        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

        z = w3.dot(fc) + b3  # first dense layer
        z[z <= 0] = 0  # pass through ReLU non-linearity

        out = w4.dot(z) + b4  # second dense layer

        probs = cls.softmax(out)  # predict class probabilities with the softmax activation function

        loss = cls.categoricalCrossEntropy(probs, label)  # categorical cross-entropy loss

        dout = probs - label  # derivative of loss w.r.t. final dense layer output
        dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
        db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases

        dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
        dz[z <= 0] = 0  # backpropagate through ReLU
        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis=1).reshape(b3.shape)

        dfc = w3.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
        dpool = dfc.reshape(pooled.shape)  # reshape fully connected into dimensions of pooling layer

        dconv2 = cls.maxpoolBackward(dpool, conv2, pool_f,pool_s)
        dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

        dconv1, df2, db2 = cls.convolutionBackward(dconv2, conv1, f2,conv_s)
        dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

        dimage, df1, db1 = cls.convolutionBackward(dconv1, image, f1,conv_s)

        grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

        return grads, loss

    @classmethod
    def adamGD(cls, batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        X = batch[:, 0:-1]  # get batch inputs
        X = X.reshape(len(batch), n_c, dim, dim)
        Y = batch[:, -1]  # get batch labels

        cost_ = 0
        batch_size = len(batch)

        # initialize gradients and momentum,RMS params
        df1 = np.zeros(f1.shape)
        df2 = np.zeros(f2.shape)
        dw3 = np.zeros(w3.shape)
        dw4 = np.zeros(w4.shape)
        db1 = np.zeros(b1.shape)
        db2 = np.zeros(b2.shape)
        db3 = np.zeros(b3.shape)
        db4 = np.zeros(b4.shape)

        v1 = np.zeros(f1.shape)
        v2 = np.zeros(f2.shape)
        v3 = np.zeros(w3.shape)
        v4 = np.zeros(w4.shape)
        bv1 = np.zeros(b1.shape)
        bv2 = np.zeros(b2.shape)
        bv3 = np.zeros(b3.shape)
        bv4 = np.zeros(b4.shape)

        s1 = np.zeros(f1.shape)
        s2 = np.zeros(f2.shape)
        s3 = np.zeros(w3.shape)
        s4 = np.zeros(w4.shape)
        bs1 = np.zeros(b1.shape)
        bs2 = np.zeros(b2.shape)
        bs3 = np.zeros(b3.shape)
        bs4 = np.zeros(b4.shape)

        for i in range(batch_size):
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

            # Collect Gradients for training example
            grads, loss = cls.conv(x, y, params, 1, 2, 2)
            [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

            df1 += df1_
            db1 += db1_
            df2 += df2_
            db2 += db2_
            dw3 += dw3_
            db3 += db3_
            dw4 += dw4_
            db4 += db4_

            cost_ += loss

        # Parameter Update

        v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
        s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
        f1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

        bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
        bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
        b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

        v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
        s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
        f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

        bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
        bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
        b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

        v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
        s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
        w3 -= lr * v3 / np.sqrt(s3 + 1e-7)

        bv3 = beta1 * bv3 + (1 - beta1) * db3 / batch_size
        bs3 = beta2 * bs3 + (1 - beta2) * (db3 / batch_size) ** 2
        b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

        v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
        s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
        w4 -= lr * v4 / np.sqrt(s4 + 1e-7)

        bv4 = beta1 * bv4 + (1 - beta1) * db4 / batch_size
        bs4 = beta2 * bs4 + (1 - beta2) * (db4 / batch_size) ** 2
        b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

        cost_ = cost_ / batch_size
        cost.append(cost_)

        params = [f1, f2, w3, w4, b1, b2, b3, b4]

        return params, cost

    @classmethod
    def categoricalCrossEntropy(cls, probs, label):
        return -np.sum(label * np.log(probs))

    @classmethod
    def softmax(cls, raw_preds):
        out = np.exp(raw_preds) # exponentiate vector of raw predictions
        return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.