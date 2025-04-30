FROM python:3.11.8

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    kafka-python \
    requests \
    beautifulsoup4 \
    polars \
    lxml

# Expose any necessary ports (if needed for Kafka communication)
EXPOSE 29092

ENV BOOTSTRAP_SERVERS=broker:29092

CMD ["python", "-u", "kafka_producer.py"]
