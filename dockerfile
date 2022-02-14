FROM python:latest
ADD skipGram.py /
ADD test_format.sh /
COPY requirements.txt requirements.txt
#COPY 1-billion-word-language-modeling-benchmark-r13output.tar.gz 1-billion-word-language-modeling-benchmark-r13output.tar.gz
COPY skipGram.tar.gz skipGram.tar.gz
RUN pip3 install -r requirements.txt
COPY . .
#SHELL ["/bin/bash", "-c", "source ./test_format.sh skipGram.tar.gz"]
#RUN ./test_format.sh skipGram.tar.gz