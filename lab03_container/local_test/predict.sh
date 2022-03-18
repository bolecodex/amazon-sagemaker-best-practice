#!/bin/bash

# 接收命令行中的第二个值作为参数
payload=$1
# 接收命令行中的第三个值作为参数, 如果为null,则默认为 【text/csv】
content=${2:-text/csv}

# 发送一个http请求
curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8080/invocations
