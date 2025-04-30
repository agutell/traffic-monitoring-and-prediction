FROM python:3.11.8

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    pandas \
    polars \
    pyarrow \
    torch \
    torchvision \
    torchaudio

CMD ["python", "-u", "model.py"]