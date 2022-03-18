#!/bin/sh

# 接收命令行中的第二个值作为参数
image=$1

# 运行docker容器
docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve
