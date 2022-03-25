#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
# 接收命令行中的第二个值作为参数， 该变量为一个指定镜像的名称，
image=$1

# 判断镜像名是否为空， 如果为空，则报错
if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# 授予对应的两个文件可以行权限
chmod +x decision_trees/train
chmod +x decision_trees/serve

# Get the account number associated with the current IAM credentials
# 获取aws账号信息
account=$(aws sts get-caller-identity --query Account --output text)

# 上一条命令返回值不等于 0, 则执行下面的语句
if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
# 获取 region 信息
region=$(aws configure get region)
region=${region:-us-west-2}

# 定义文件名
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
# 检查ECR的repository中是否有对应的镜像信息
"""
/dev/null 代表空设备文件 
> 代表重定向到哪里，例如：echo "123" > /home/123.txt 
1 表示stdout标准输出，系统默认值是1，所以">/dev/null"等同于"1>/dev/null" 
2 表示stderr标准错误 
& 表示等同于的意思，2>&1，表示2的输出重定向等同于1 

"""
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

# 上一条命令返回值不等于-ne 0, 则执行下面的语句
if [ $? -ne 0 ]
then
    # 在ecr中创建一个 repository
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it 
# 登录到所属region的ECR
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# build 一个镜像。
docker build  -t ${image} .
# 设置镜像 tag
docker tag ${image} ${fullname}
# push 镜像
docker push ${fullname}
