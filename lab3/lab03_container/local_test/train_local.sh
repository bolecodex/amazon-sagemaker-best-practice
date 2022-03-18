#!/bin/sh

# 接收命令行中的第二个值作为参数
image=$1

# 创建目录
mkdir -p test_dir/model
mkdir -p test_dir/output

# 删除目录中的所有内容
rm test_dir/model/*
rm test_dir/output/*

# 运行docker容器
docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
