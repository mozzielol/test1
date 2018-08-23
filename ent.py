import numpy as np

def get_gradients(model):
    '''
    Return the gradient of every trainable weight in model
    '''
    weights = [tensor for tensor in model.trainable_weights]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)


def calc_ent(x):
    '''
    calculate shanno entropy of x
    '''

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calc_condition_ent(x, y):
    '''
    calculate ent H(y|x)
    '''

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent

def calc_ent_grap(x,y):
    '''
    calculate ent grap
    '''
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


def batch_diff(pre,cur,batch_size=128):

    diff = []
    for (d1,d2) in zip(pre.reshape(batch_size,-1),cur.reshape(batch_size,-1)):
        diff.append(calc_ent_grap(d1,d2))

    return sum(diff)

def batch_loss(diff_t,diff_c,batch_size=128):
    d = []
    for diff in diff_t:
        d.append(batch_diff(diff,diff_c))
    
    return d.index(min(d))

def Data_loss(X,batch):
    e = []
    for data in X:
        e.append(entropy(data,b).sum())
    return min(e)

