import tensorflow as tf
import argparse
import os
import numpy as np
import json


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""
    # Sequential： 创建顺序模型, 此模型为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。 参数：数组中的内容为模型中的层次结构
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),                                  # Flatten层用来将输入“压平”，即把多维的输入一维化处理
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),         # 该层网络有1024个神经元， 激励函数为： relu
        tf.keras.layers.Dropout(0.4),                               # 利用Dropout函数防止过拟合，  表示将有多少神经元暂时从网络中丢弃
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)         # 该层网络有10个神经元， 激励函数为： softmax
    ])

    # 对模型进行compile操作， 使用的优化器为 adam, 损失函数为： sparse_categorical_crossentropy， 监控指标为 accuracy
    # adam 优化器： 是对SGD的扩展, 实现简单，计算高效，对内存需求少.自动调整学习率,很适合应用于大规模的数据及参数的场景.适用于梯度稀疏或梯度存在很大噪声的问题
    # sparse_categorical_crossentropy 损失函数： categorical_crossentropy和sparse_categorical_crossentropy都是计算多分类crossentropy的，只是对y的格式要求不同。
        # 1）如果是categorical_crossentropy，那y必须是one-hot处理过的
        # 2）如果是sparse_categorical_crossentropy，那y就是原始的整数形式.
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 训练数据              
    model.fit(x_train, y_train)

    # 验证数据
    model.evaluate(x_test, y_test)

    return model


def _load_training_data(base_dir):
    """Load MNIST training data"""
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
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


if __name__ == "__main__":
    args, unknown = _parse_args()

    # 加载训练数据和测试数据
    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    # 创建模型
    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        # 将模型进行导出， 默认格式为 h5 形式、
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000001'))
