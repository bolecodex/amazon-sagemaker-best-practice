# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

# 获取基础镜像
FROM public.ecr.aws/lts/ubuntu:18.04

# 作者信息
MAINTAINER Amazon AI <sage-learner@amazon.com>

# 安装相应的软件
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
         build-essential \
         file \
         python-dev \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
# 安装pip， 并安装相应的lib库
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && python get-pip.py && \
    pip install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask gevent gunicorn && \
        (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

# 设置python环境变量
# 关于日志输出
ENV PYTHONUNBUFFERED=TRUE
# 关于字节码
ENV PYTHONDONTWRITEBYTECODE=TRUE
# 设置环境变量目录
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
# copy文件
COPY decision_trees /opt/program
# 指定工作目录
WORKDIR /opt/program

