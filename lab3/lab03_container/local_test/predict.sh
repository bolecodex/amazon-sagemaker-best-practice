#!/bin/bash

# 接收命令行中的第二个值作为参数
payload=$1
# 接收命令行中的第三个值作为参数, 如果为null,则默认为 【text/csv】
content=${2:-text/csv}

# 发送一个http请求
# 参数：http://aiezu.com/article/linux_curl_command.html
# --data-binary：HTTP方式POST二进制数据；
# 如果数据以“@”开头，后紧跟一个文件，将post文件内的内容；
# -v: 显示更详细的信息，调试时使用；
curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8080/invocations
