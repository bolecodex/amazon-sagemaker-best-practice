# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Evaluation script for measuring model accuracy."""

import json
import os
import tarfile
import logging
import pickle

import pandas as pd
import xgboost

# 初始化日志对象
logger = logging.getLogger()
# 设置日志的级别， 为INFO
logger.setLevel(logging.INFO)
# 将指定的StreamHandler处理器添加到记录器.
logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


if __name__ == "__main__":
    # 设置文件路径
    model_path = "/opt/ml/processing/model/model.tar.gz"
    # 打开压缩文件
    with tarfile.open(model_path) as tar:
        # 解压文件
        tar.extractall(path="..")
    # 打印日志
    logger.debug("Loading xgboost model.")
    # pickle.load： 加载数据， 将文件中的数据解析为一个Python对象
    # open： 以可读2进制的方式读取模型文件
    model = pickle.load(open("xgboost-model", "rb"))

    print("Loading test input data")
    # 定义路径
    test_path = "/opt/ml/processing/test/test.csv"
    # 读取csv数据,不带 header
    df = pd.read_csv(test_path, header=None)
    logger.debug("Reading test data.")
    # df.iloc: 基于行、列索引（index、columns）进行索引查询，
    # to_numpy(): 将结果转成nuympy数组
    y_test = df.iloc[:, 0].to_numpy()
    # 在原数据上直接除去第0列内容。inplace=True，表示在原有数据上直接操作。
    df.drop(df.columns[0], axis=1, inplace=True)
    # xgboost.DMatrix来读取数据
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    # 进行预测
    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    # accuracy_score: 获取分类准确率， 两个参数分别为： 真实值 和 预测值。
    # round()： 对数值四舍五入。
    acc = accuracy_score(y_test, predictions.round())
    # roc_auc_score： 根据预测结果操作特征曲线 (ROC AUC) 下的面积。
    auc = roc_auc_score(y_test, predictions.round())

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    # 定义 report_dict， 其中 standard_deviation标准差为NaN
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    print("Classification report:\n{}".format(report_dict))
    # 定义路径
    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    print("Saving classification report to {}".format(evaluation_output_path))

    # 打开一个文件写入操作流
    with open(evaluation_output_path, "w") as f:
        # 写入JOSN对象写入文件
        f.write(json.dumps(report_dict))
