# **Real-time traffic monitoring and prediction of traffic flow**
### A project in machine learning and intelligent systems.
![Demo](assets/traffic.gif)
#### *Authors: Dan Vicente (dan.vicente.ihanus@gmail.com), Alexander Gutell (alex.gutell@gmail.com)*

This project was developed to gain experience in Dockerized applications and analysis of real-time streaming time series data. We built an end-to-end application that:

* Collects real-time traffic data from Trafikverket's API through a Kafka producer
* Processes streaming data using Apache Kafka
* Performs real-time analysis and feature engineering with PySpark
* Forecasts traffic conditions five minutes into the future using:
  * An autoregressive Gated Recurrent Unit (GRU) built using PyTorch
  * An autoregressive Moving Average model for prediction and comparison

Visualizes both current traffic conditions and predictions in an interactive dashboard


## Application details

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="assets/kafka.png" alt="Kafka Logo" width="120"/>
  <img src="assets/spark.png" alt="Spark Logo" width="120"/>
  <img src="assets/docker-logo-blue.png" alt="Docker Logo" width="120"/>
</div>
The application follows a microservices architecture with several components:

Data Producer: Python service that queries Trafikverket's API and sends data to Kafka
Kafka Broker: Message queue that ensures reliable data delivery between services
Data Consumer: Processes and transforms incoming data streams
ML Model Service: GRU for traffic prediction
Visualization Service: Streamlit dashboard for real-time monitoring

All components are containerized with Docker and orchestrated with Docker Compose, allowing for easy scaling and deployment. The data flows through the system as follows:
Trafikverket API → Producer → Kafka → Consumer → PySpark Processing → ML Prediction → Dashboard

This architecture enables real-time data processing with minimal latency while maintaining system reliability.
## Setup

### Clone the Repository
First, clone this repository to your local machine:
Make sure you have Docker and Docker Compose installed on your system before proceeding.
### Environment Variables
This application requires a `.env` file with API credentials and configuration parameters. 

Create one before starting:

1. Create the `.env` file:
```bash
touch .env
```

2. Add the required parameters to the file:
```bash
# Trafikverket API key
echo "TRAFIKVERKET_API_KEY=your_api_key_here" >> .env

# Kafka config
echo "KAFKA_BOOTSTRAP_SERVERS=broker:29092" >> .env
echo "KAFKA_TOPIC=traffic_data" >> .env

# region and samples config
echo "REGION_ID=4" >> .env
echo "N_SAMPLES=100000" >> .env
```

**Note:** Replace `your_api_key_here` with your actual Trafikverket API key. You can get an API key free of charge by registering at [Trafikverket's API Portal](https://api.trafikinfo.trafikverket.se/).

## How to run the application

1. Install Docker and Docker Compose
2. Create the `.env` file as described above
3. Run the application:
```bash
docker-compose up --build
```
4. Go to http://localhost:8501/ in your browser
5. Wait 20 minutes until the model starts monitoring and forecasting, the model waits for data the first 20 minutes

## Project Structure
- `producer/`: Contains code for collecting data from Trafikverket API
- `consumer/`: Contains code for processing the Kafka stream
- `models/`: Contains the models for traffic prediction
- `visualization/`: Contains the Streamlit dashboard

<!-- ## NOTE: All of this will maybe (probably?) be changed into a single docker container

To get started:

1. Install uv-astral: 

```
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

2. Clone the repository:
```bash
$ git clone "this-url"
$ cd "repo-dir"
```

3. Create and activate a virtual environment:
```bash
$ python -m venv .venv
$ source .venv/bin/activate  
# On Windows use: 
$ source .venv\Scripts\activate
```

4. Install dependencies using `uv-astral`:
```bash
$ uv sync
```

5. Verify the setup by running the application:
```bash
$ python hello.py
```
## Running the scripts
    - More stuff coming here


## To see the package and dependencies 
```bash
$ cat ./pyproject.toml
```

## How to Start Producer from Terminal
In the terminal, navigate to the directory with the Dockerfile (cd pytorch) and run:
```docker build -t my-kafka-producer .```

After building the image, you can run it using the following command:
```docker run --network="host" my-kafka-producer```


## References -->
