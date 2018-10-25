import numpy as np
import keras.backend as K



def get_weights(model):

    return model.get_weights()


def get_gradients(model):
    '''
    Return the gradient of every trainable weight in model
    '''
    weights = [tensor for tensor in model.trainable_weights]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from skimage.util import random_noise

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from copy import deepcopy
def create_permuted_mnist_task(num_datasets):
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    task_list = [mnist]
    for seed in range(1, num_datasets):
        task_list.append(permute(mnist, seed))
    return task_list

def permute(task, seed):
    np.random.seed(seed)
    perm = np.random.permutation(task.train._images.shape[1])
    permuted = deepcopy(task)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]
    return permuted






