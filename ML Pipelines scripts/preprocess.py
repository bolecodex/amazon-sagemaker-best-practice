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
"""Feature engineers the customer churn dataset."""
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

# 初始化日志对象
logger = logging.getLogger()
# 设置日志的级别， 为INFO
logger.setLevel(logging.INFO)
# 将指定的StreamHandler处理器添加到记录器.
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    # 创建一个参数解析对象
    parser = argparse.ArgumentParser()
    # 定义一个运行必要参数
    parser.add_argument("--input-data", type=str, required=True)
    # 解析输入的参数
    args = parser.parse_args()

    # 定义一个路径
    base_dir = "/opt/ml/processing"
    # 创建一个路径。 mkdir: parents=True自动创建不存在的目录， exist_ok=True如果存在则忽略
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    # 获取 input_data 参数
    input_data = args.input_data
    print(input_data)
    # 分割数据
    bucket = input_data.split("/")[2]
    # 拼接 key 
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    # 定义文件路径
    fn = f"{base_dir}/data/raw-data.csv"
    # 创建一个S3对象
    s3 = boto3.resource("s3")
    # 下载文件 
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")

    # read in csv
    # 读取csv
    df = pd.read_csv(fn)

    # drop the "Phone" feature column
    # 除去数据中的 Phone 列数据
    df = df.drop(["Phone"], axis=1)

    # Change the data type of "Area Code"
    # 将数据中的 Area Code 列转为对象
    df["Area Code"] = df["Area Code"].astype(object)

    # Drop several other columns
    # 除去 "Day Charge", "Eve Charge", "Night Charge", "Intl Charge" 列数据
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # Convert categorical variables into dummy/indicator variables.
    # 将数据进行 one hot encode操作
    model_data = pd.get_dummies(df)

    # Create one binary classification target column
    # 将数据进行连接
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )

    # Split the data
    # 将数据集按一定的比例化分为 train_data, validation_data, test_data。
    # np.split： 分割数据。 参数1： n维数组， 参数2： 切分区间， 参数3： 表示的是沿哪个维度切，默认为0表示横向切，为1时表示纵向切
    # sample: 随机选取若干行, 参数1： frac表示抽取行的比例， random_state：随机因子
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    # 将 train_data 写入CSV,不带index 和 header
    pd.DataFrame(train_data).to_csv(
        f"{base_dir}/train/train.csv", header=False, index=False
    )
    # 将 validation_data 写入CSV,不带index 和 header
    pd.DataFrame(validation_data).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    # 将 test_data 写入CSV,不带index 和 header
    pd.DataFrame(test_data).to_csv(
        f"{base_dir}/test/test.csv", header=False, index=False
    )
