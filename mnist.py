# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import json
import argparse
from tensorflow.python.platform import tf_logging
import logging as _logging
import sys as _sys


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    #If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. 
    #In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
    # https://www.tensorflow.org/api_docs/python/tf/reshape

    # MNIST数据集是一个训练集一共包含了 60,000 张图像和标签，而测试集一共包含了 10,000 张图像和标签的手写数字数据集。
    # 每张图 28x28 pixels, 颜色为黑白。
    # 利用 reshape 函数将数据转换成 4d张量. batch_size = -1,表示对全数据进行转换,width 和 height为28, channels为1表示黑白， 3为RGB彩色。
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]

    # 对 input_layer 进行卷积运算
    # filters： 简单理解就是一个filter表示图像的一层， 32个filter就是有32层。
    # kernel_size： 表示参与卷积运算的卷积核大小为 5 * 5的矩阵
    # padding的方式为： same padding： 当filter的中心点与image的边角重合时，开始做卷积运算。
                    # valid padding： 当filter的中心点与image的边角重合时，开始做卷积运算
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]

    # 将卷积输出的 feature map， 通过 2 * 2的pool_size，进行 max_pooling操作， 每次移动步长为2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    
    # 将卷积运算输出的结果从多维度转成一维
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    # 设置全连接层信息， 输出神经元个数为 1024， 激励函数为 relu
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    # 添加dropout操作，为了防止过拟合。
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    # 定义最后一层网络结构,输出的神经元个数为 10
    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`. 
        """
        softmax 函数： softmax函数，又称归一化指数函数。是二分类函数sigmoid在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。
        概率有两个性质：1）预测的概率为非负数；2）各种预测结果概率之和等于1
        详细参考PPT: https://docs.google.com/presentation/d/1tBDnnWs2KxzEombv1j3luhMRZ2Z1KihAhaaFq1Y8Y8A/edit?usp=sharing
        """

        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # EstimatorSpec: 
    """
        EstimatorSpec一个class(类)，是定义在model_fn中的，并且model_fn返回的也是它的一个实例，这个实例是用来初始化Estimator类的.
        根据mode的值的不同,需要不同的参数：
        * 对于mode == ModeKeys.TRAIN：必填字段是loss和train_op.
        * 对于mode == ModeKeys.EVAL：必填字段是loss.
        * 对于mode == ModeKeys.PREDICT：必填字段是predictions.
    """
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # 定义损失函数为 sparse_softmax_cross_entropy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    # 定义训练优化器 GradientDescent， 学习率为：  0.001
    # global_step: 是一个Variable类型的参数，在所有的网络参数结束梯度更新后，global_step会自增加一。
    #               使用global_step作为梯度更新次数控制整个训练过程何时停止，就相当于使用迭代次数（num of iterations）作为控制条件。
    #               在一次迭代过程，就前向传播了一个batch，并计算之后更新了一次梯度。
    # tf.train.get_global_step() 方法返回的是的 global_step作为name的tensor, 
    #   如 <tf.Variable ‘global_step:0’ shape=() dtype=int64_ref>。 tensor参数与global_step = tf.Variable(0, name=“global_step”, trainable=False) 完全相同。
    if mode == tf.estimator.ModeKeys.TRAIN: 
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train

def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    return x_test, y_test

def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

def serving_input_fn():
#     tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
#     服务输入函数的工作是将接收到的原始特征转换为模型函数接受的已处理特征。
#     receiver_tensors：这些是输入占位符。这将在您的图表中打开，您将在其中接收原始输入特征。
#     定义此占位符后，您可以对这些接收器张量执行转换，以将它们转换为模型可接受的特征。
    inputs = {'x': tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    # Create the Estimator
    """
        Estimator: 是TensorFlow提供的高阶API Class，Estimator 会封装下列操作：训练，评估，预测，导出以供使用。
        初始化函数：
            __init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
            参数：
            model_fn: 模型函数。函数的格式如下：
                参数：
                    1、features: 这是 input_fn 返回的第一项（input_fn 是 train, evaluate 和 predict 的参数）。类型应该是单一的 Tensor 或者 dict。
                    2、labels: 这是 input_fn 返回的第二项。类型应该是单一的 Tensor 或者 dict。如果 mode 为 ModeKeys.PREDICT，则会默认为 labels=None。如果 model_fn 不接受 mode，model_fn 应该仍然可以处理 labels=None。
                    3、mode: 可选。指定是训练、验证还是测试。参见 ModeKeys。
                    4、params: 可选，超参数的 dict。 可以从超参数调整中配置 Estimators。
                    5、config: 可选，配置。如果没有传则为默认值。可以根据 num_ps_replicas 或 model_dir 等配置更新 model_fn。
                返回：
                    EstimatorSpec
            model_dir: 保存模型参数、图等的地址，也可以用来将路径中的检查点加载至 estimator 中来继续训练之前保存的模型。如果是 PathLike， 那么路径就固定为它了。如果是 None，那么 config 中的 model_dir 会被使用（如果设置了的话），如果两个都设置了，那么必须相同；如果两个都是 None，则会使用临时目录。
    """
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=args.model_dir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    
    # LoggingTensorHook： 监视/记录张量， 用标签“probabilities”记录“Softmax”张量中的值
    # tf.estimator.LoggingTensorHook(tensors, every_n_iter=None, every_n_secs=None, at_end=False, formatter=None)
    # tensors： 参数类型dic 将字符串值标签映射到张量/张量名称或 iterable 的张量/张量名称的字典。
    # every_n_iter： 参数类型int ，在当前worker上每N个局部步长打印一次 tensors 量值。
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    # 创建一个已经经过验证的TrainSpec实例.用来配置训练过程
    # tf.estimator.TrainSpec函数返回一个经过验证的TrainSpec对象.
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=20000)
    # 创建一个已经验证的EvalSpec实例. 用来配置评估过程、（可选）模型的导出。
    # tf.estimator.EvalSpec函数返回一个经过验证的EvalSpec对象.
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)
    # 进行训练和评估
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
    if args.current_host == args.hosts[0]:
        mnist_classifier.export_savedmodel(args.sm_model_dir, serving_input_fn)
        
    # The tf.estimator.Estimator interface allows users to provide a model_fn which accepts features either within a single tensor or within a dictionary mapping strings to tensors.

    # The Estimator export_savedmodel method requires a serving_input_receiver_fn argument, which is a function of no arguments that produces a ServingInputReceiver. The features tensors from this ServingInputReceiver are passed to the model_fn for serving.

    # Upon instantiation, the ServingInputReceiver wraps single tensor features into a dictionary. This raises an error for estimators whose model_fn expects a single tensor as its features argument.