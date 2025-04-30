FROM openjdk:8-jdk-slim

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip curl && \
    pip3 install pyspark kafka-python requests beautifulsoup4 polars && \
    curl -LO https://archive.apache.org/dist/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3.tgz && \
    tar -xvzf spark-3.5.2-bin-hadoop3.tgz && \
    mv spark-3.5.2-bin-hadoop3 /usr/local/spark && \
    rm spark-3.5.2-bin-hadoop3.tgz

ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV SPARK_JARS_PACKAGES=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2

COPY . /app

CMD ["python3", "-u", "spark_consumer.py"]



