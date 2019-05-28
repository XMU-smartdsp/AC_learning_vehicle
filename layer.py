import tensorflow as tf
from tensorflow.contrib.layers import batch_norm as bn


def conv2d(name, input, conv_size, out_num, decay=(0.0005,0),strides=(1,1,1,1),pad='VALID',active = 'relu',Init=None):
    h = conv_size[0]
    w = conv_size[1]
    in_num = input.get_shape().as_list()[-1]
    if Init==None:
        conv_w = tf.get_variable(initializer=tf.random_normal([h, w, in_num, out_num],stddev=0.005),name=name+'_weight')
        conv_b = tf.get_variable(initializer=tf.random_normal([out_num]),name=name+'_bias')
    else:
        #print (h, w, in_num, out_num),Init[0].shape
        assert((h, w, in_num, out_num)==Init['weights'].shape)
        conv_w = tf.get_variable(initializer=Init['weights'],name = name+'_weight')
        conv_b = tf.get_variable(initializer=Init['biases'],name = name+'_bias')
    '''penalty term'''
    w_decay = tf.multiply(tf.nn.l2_loss(conv_w), decay[0], name= name +'_weight_regularizer')
    b_decay = tf.multiply(tf.nn.l2_loss(conv_b), decay[1], name= name +'_bias_regularizer')
    tf.add_to_collection('losses', w_decay)
    tf.add_to_collection('losses', b_decay)

    out = tf.nn.bias_add(tf.nn.conv2d(input, conv_w, strides=strides, padding=pad), conv_b)
    if active =='relu':
        return tf.nn.relu(out,name=name+'_output')
    elif active == None:
        return out
    else:
        print('wrong')
        exit()

def conv2d_with_group(name, input, conv_size, out_num, group, pad = 'VALID', decay=(0.0005,0),strides=(1,1,1,1),active = 'relu',Init=None):
    h = conv_size[0]
    w = conv_size[1]
    in_num = int(input.get_shape().as_list()[-1]/group)

    if Init==None:
        conv_w = tf.get_variable(initializer=tf.random_normal([h, w, in_num, out_num],stddev=0.005),name=name+'_weight')
        conv_b =  tf.get_variable(initializer=tf.random_normal([out_num]),name=name+'_bias')
    else:
#        print (h, w, in_num, out_num),Init['weights'].shape
        assert((h, w, in_num, out_num)==Init['weights'].shape)
        conv_w = tf.get_variable(initializer=Init['weights'],name = name+'_weight')
        conv_b = tf.get_variable(initializer=Init['biases'],name = name+'_bias')

    '''penalty term'''
    w_decay = tf.multiply(tf.nn.l2_loss(conv_w), decay[0], name= name +'_weight_regularizer')
    b_decay = tf.multiply(tf.nn.l2_loss(conv_b), decay[1], name= name +'_bias_regularizer')
    tf.add_to_collection('losses', w_decay)
    tf.add_to_collection('losses', b_decay)

    conv_groups = tf.split(input,group,3)
    weights_groups = tf.split(conv_w, group, 3)

    conv = [tf.nn.conv2d(i, k, strides, padding=pad) for i, k in zip(conv_groups, weights_groups)]
    conv = tf.concat(conv,3)
    out = tf.nn.bias_add(conv, conv_b)
    if active =='relu':
        return tf.nn.relu(out,name=name+'_output')
    elif active == None:
        return out
    else:
        print('wrong')
        exit()

def conv3d(name,input,conv_size,out_num,decay=(0.0005,0),strides=(1,1,1,1,1),padding='VALID',active = 'relu'):
   

    h = conv_size[1]
    w = conv_size[2]
    in_num = input.get_shape().as_list()[-1]
    conv_w = tf.get_variable(initializer=tf.random_normal([conv_size[0],h, w, in_num, out_num],stddev=0.005),name=name+'_weight')
    conv_b = tf.get_variable(initializer=tf.random_normal([out_num]),name=name+'_bias')
 
    '''penalty term'''
    w_decay = tf.multiply(tf.nn.l2_loss(conv_w), decay[0], name= name +'_weight_regularizer')
    b_decay = tf.multiply(tf.nn.l2_loss(conv_b), decay[1], name= name +'_bias_regularizer')
    tf.add_to_collection('losses', w_decay)
    tf.add_to_collection('losses', b_decay)

    out = tf.nn.bias_add(tf.nn.conv3d(input,conv_w,strides=strides,padding = padding),conv_b)
    if active =='relu':
        return tf.nn.relu(out,name=name+'_output')
    elif active == None:
        return out
    else:
        print('wrong')
        exit()

def fully_connect(name, input, out_num, decay=(0.0005,0), active = 'relu',Init = None):
    
    in_num = input.get_shape().as_list()[-1]
    if Init == None:
        fc_w = tf.get_variable(initializer=tf.random_normal([in_num,out_num],stddev=0.005),name=name+'_weight')
        fc_b = tf.get_variable(initializer=tf.random_normal([out_num]),name=name+'_bias')
    else:
        assert(in_num==Init['weights'].shape[0])
        fc_w = tf.get_variable(initializer=Init['weights'],name = name+'_weight')
        fc_b = tf.get_variable(initializer=Init['biases'],name = name+'_bias')
    '''penalty term'''
    w_decay = tf.multiply(tf.nn.l2_loss(fc_w), decay[0], name= name +'_weight_regularizer')
    b_decay = tf.multiply(tf.nn.l2_loss(fc_b), decay[1], name= name +'_bias_regularizer')
    tf.add_to_collection('losses', w_decay)
    tf.add_to_collection('losses', b_decay)

    out = tf.matmul(input, fc_w) + fc_b
    if active =='relu':
        return tf.nn.relu(out,name=name+'_output')
    elif active == None:
        return out
    else:
        print('wrong')
        exit()

def max_pooling(name,input,poolsize,stride):
    return tf.nn.max_pool(input, ksize=[1, poolsize, poolsize, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

def max_pooling3d(name,input,poolsize,stride):
    return tf.nn.max_pool3d(input,ksize=[1,1,poolsize,poolsize,1],strides=[1,1,stride,stride,1],padding='VALID', name=name)

def batch_normalize(input,phase):
    return bn(input,scale=True,updates_collections=None,is_training=phase)


def dropout(name,input,keep_prob):
    return tf.nn.dropout(input,keep_prob,name=name)


def global_pooling(name,input):
    h = input.get_shape().as_list()[1]
    w = input.get_shape().as_list()[2]
    return tf.nn.avg_pool(input,[1,h,w,1],strides=[1,1,1,1],padding='VALID')


def global_maxpooling(name,input):
    h = input.get_shape().as_list()[1]
    w = input.get_shape().as_list()[2]
    return tf.nn.max_pool(input,[1,h,w,1],strides=[1,1,1,1],padding='VALID')

def global_maxpooling3d(name,input):
    dim_1 = input.get_shape().as_list()[1]
    dim_2 = input.get_shape().as_list()[2]
    dim_3 = input.get_shape().as_list()[3]
    return tf.nn.max_pool3d(input,ksize=[1,dim_1,dim_2,dim_3,1],strides=[1,1,1,1,1],padding='VALID',name=name) 
