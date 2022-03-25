# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    # 定义一个get_model类方法
    @classmethod
    def get_model(cls):   # new ScoringService().get_model()    ScoringService.get_model()
        """Get the model object for this instance, loading it if it's not already loaded."""
        # 判断model是否为None
        if cls.model == None:
            # 打开路径下的文件
            with open(os.path.join(model_path, 'decision-tree-model.pkl'), 'r') as inp:
                # 将文件流反序列化成python对象 
                cls.model = pickle.load(inp)
        # 返回模型        
        return cls.model

    # 定义一个predict类方法
    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        # 获取模型        
        clf = cls.get_model()
        # 返回预测结果
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

# 定义一个服务器的GET请求
@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    
    # 判断模型是否存在，如果存在返回200，否则返回404
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

# 定义一个服务器的POST请求
@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    # 判断请求的 content_type 是否为 'text/csv'
    if flask.request.content_type == 'text/csv':
        # 获取请求数据
        data = flask.request.data.decode('utf-8')
        # 将数据加载到内存当中
        s = StringIO.StringIO(data)
        # 读取csv文件
        data = pd.read_csv(s, header=None)
    else:
        # 返回一个415的错误响应
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    # 进行预测
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    # 生成一个字符串操作流
    out = StringIO.StringIO()
    # 将数据写入到csv中
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    # 获取写入的字符串信息
    result = out.getvalue()

    # 返回最终结果
    return flask.Response(response=result, status=200, mimetype='text/csv')
