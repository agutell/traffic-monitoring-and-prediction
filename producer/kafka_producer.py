import requests
from bs4 import BeautifulSoup
import polars as pl
from datetime import datetime, timedelta
from kafka.errors import KafkaError
import time
import sys
import json
from kafka import KafkaProducer
import os

def trafikverket_post_query(
    api_key: str,
    producer: KafkaProducer,
    kafka_topic: str,
    object_type: str = "TrafficFlow", 
    region_id: int = 4,
    site_id: str = None,
    version: str = "1.5",
    n_samples: int = 1000
) -> None:
    
    """ 
    Perform a XML query to the trafikverket site and send the data to a Kafka topic
    
    Args
    ------ 
        api_key: your api_key,

        producer: Kafka producer
        
        kafka_topic: Kafka topic to send the data to
        
        object_type: The variable of interest (default = TrafficFlow), 
        
        region_id: Region to consider (default = 4 = Stockholm)
        
        site_id: Which measurement site to consider (hardcoded at the moment)
        
        version: Which api-version to use (default = 1.5, what we need for TrafficFlow)
        
        n_samples: How many samples to obtain
    
        
    Return
    ------
        None
    """
    url = "https://api.trafikinfo.trafikverket.se/v2/data.xml"
    limit = str(n_samples)
    
    flag = True
    while flag:
    # Calculate a time range for the last 1 minute
        #end_time = datetime.now().replace(second=0, microsecond=0)
        #end_time = (datetime.now() - timedelta(minutes=1)).replace(second=0, microsecond=0)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=20)
        print('start time: ', start_time)
        print('end time ', end_time)
        if end_time.second < 30:
            time.sleep(5)
        else: 
            flag = False

    #start_time = end_time - timedelta(minutes=1)

    
    filter_xml = f"""
                <AND>
                <GT name="MeasurementTime" value="{start_time.isoformat()}Z" />
                <LT name="MeasurementTime" value="{end_time.isoformat()}Z" />
                """
    if region_id is not None:
        filter_xml += f'<EQ name="RegionId" value="{region_id}" />'
    if site_id is not None:
        filter_xml += f'<EQ name="SiteId" value="{site_id}" />' 
    filter_xml += "</AND>"

    headers = {
        'Content-Type': 'application/xml'
    }
    xml_payload = f"""
    <REQUEST>
        <LOGIN authenticationkey="{api_key}" />
        <QUERY objecttype="{object_type}" schemaversion="{version}" limit="{limit}">
            <FILTER>
                {filter_xml}
            </FILTER>
        </QUERY>
    </REQUEST>
    """
    
    
    r = requests.post(url=url, 
                      data=xml_payload, 
                      headers=headers)
    
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, "lxml-xml")
        
        data_arr = []
        for traffic_flow in soup.find_all("TrafficFlow"):
            row = {}
            for child in traffic_flow.find_all(recursive=False):
                row[child.name] = child.text
            
            # Convert datatypes
            row['SiteId'] = int(row['SiteId'])
            row['VehicleFlowRate'] = int(row['VehicleFlowRate'])
            row['AverageVehicleSpeed'] = float(row['AverageVehicleSpeed'])
            row['MeasurementTime'] = datetime.fromisoformat(row['MeasurementTime']).isoformat()
            row['ModifiedTime'] = datetime.fromisoformat(row['ModifiedTime']).isoformat()

            # Rename keys
            row['specific_lane'] = row.pop('SpecificLane')
            row['geometry'] = row.pop('Geometry')
            row['data_quality'] = row.pop('DataQuality')
            row['vehicle_flow_rate'] = row.pop('VehicleFlowRate')
            row['average_vehicle_speed'] = row.pop('AverageVehicleSpeed')
            row['measurement_time'] = row.pop('MeasurementTime')
            row['modified_time'] = row.pop('ModifiedTime')
            row['site_id'] = row.pop('SiteId')

            # Remove unnecessary fields
            row.pop('RegionId', None)
            row.pop('MeasurementOrCalculationPeriod', None)
            row.pop('VehicleType', None)
            row.pop('MeasurementSide', None)
            row.pop('Deleted', None)
            row.pop('CountyNo', None)

            try:
                producer.send(kafka_topic, value=row)
            except KafkaError as e:
                print(f"Failed to send message to Kafka: {e}")
            data_arr.append(row)
        producer.flush()
        data = pl.DataFrame(data_arr)
        print(f"Data of shape {data.shape} sent to Kafka topic: {kafka_topic}")
    else:
        print(f"Failed to retrieve data: {r.status_code}")

if __name__ == "__main__":
    API_KEY = os.environ.get("TRAFIKVERKET_API_KEY")
    OBJECT_TYPE = os.environ.get("OBJECT_TYPE", "TrafficFlow")
    VERSION = os.environ.get("VERSION", "1.5")
    N_SAMPLES = int(os.environ.get("N_SAMPLES", 100000))
    REGION_ID = int(os.environ.get("REGION_ID", 4))
    SITE_ID = os.environ.get("SITE_ID")
    KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")
    KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "traffic_data") 

    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        api_version=(0,11,5),
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    while True:
        try:
            trafikverket_post_query(api_key=API_KEY,
                                    producer=producer,
                                    kafka_topic=KAFKA_TOPIC,
                                    object_type=OBJECT_TYPE,
                                    region_id=REGION_ID,
                                    site_id=SITE_ID,
                                    version=VERSION,
                                    n_samples=N_SAMPLES)
        except Exception as e:
            print(f"{e} ocurred..")
            sys.exit()
        time.sleep(60)