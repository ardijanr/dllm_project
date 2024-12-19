FROM nvcr.io/nvidia/tensorflow:24.11-tf2-py3

RUN pip3 install transformers tf-keras huggingface-hub torch
WORKDIR /app


CMD ["python","-u", "native.py"]
