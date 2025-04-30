import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import glob
import time
import os
import logging
from datetime import datetime

# --------------- #
#   GLOBAL VARS   #
# --------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Traffic Flow Real-time Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

d = {
    4472: ['59.34447', '18.026217'],
    4195: ['59.352142', '18.048351'],
    1051: ['59.35464', '18.03645']
}

# ----- #
# PLOTS #
# ----- #

def create_map(site_id=None):
    """Create folium map with traffic flow visualization for a specific site"""
    sthlm_coords = [59.33094171272806, 18.07199652281146]
    m = folium.Map(location=sthlm_coords, zoom_start=10)
    if site_id is not None and site_id in d:
        coords = d[site_id]
        lat, lon = float(coords[0]), float(coords[1])
        folium.Marker(location=[lat, lon], popup=f"Site {site_id}").add_to(m)
        m.location = [lat, lon]
        m.zoom_start = 13
    else:
        for site_id, coords in d.items():
            lat, lon = float(coords[0]), float(coords[1])
            folium.Marker(location=[lat, lon], popup=f"Site {site_id}").add_to(m)
    return m


def create_traffic_plot(df, site_id):
    """Create traffic plot for a specific site"""
    try:
        site_data = (df
            .filter(pl.col("site_id") == site_id)
            .select([
                "measurement_time",
                "vehicle_flow_rate",
                "type",
                "forecast_type"  
            ])
            .sort("measurement_time"))
        
        if site_data.height == 0:
            logger.warning(f"No data found for site {site_id}")
            return go.Figure()
        
        fig = go.Figure()
        
        # Plot actual data
        actual_data = site_data.filter(pl.col("type") == "Actual")
        if actual_data.height > 0:
            fig.add_trace(go.Scatter(
                x=actual_data.select("measurement_time").to_series().to_list(),
                y=actual_data.select("vehicle_flow_rate").to_series().to_list(),
                mode="lines+markers",
                name="Actual",
                line=dict(width=2, color='blue'),
                marker=dict(size=8)
            ))
        
        # Plot prediction data by forecast_type
        prediction_data = site_data.filter(pl.col("type") == "Prediction")
        
        # Get unique forecast types (e.g., gru, moving_avg)
        forecast_types = prediction_data.select("forecast_type").unique().to_series().to_list()
        
        colors = {'gru': 'red', 'moving_avg': 'green'}  # Define colors for different models
        
        for forecast_type in forecast_types:
            type_data = prediction_data.filter(pl.col("forecast_type") == forecast_type)
            if type_data.height > 0:
                color = colors.get(forecast_type, 'orange')  # Default to orange if type not in dict
                name = "GRU" if forecast_type == "gru" else "Moving Avg" if forecast_type == "moving_avg" else forecast_type
                
                fig.add_trace(go.Scatter(
                    x=type_data.select("measurement_time").to_series().to_list(),
                    y=type_data.select("vehicle_flow_rate").to_series().to_list(),
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2, color=color),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=f"Traffic Flow: Site {site_id}",
            height=400,
            xaxis_title="Time",
            yaxis_title="Vehicle Flow Rate",
            legend_title="Data Type",
            template="plotly_white",
            hovermode='x unified',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating traffic plot for site {site_id}: {str(e)}")
        return go.Figure()


def create_loss_plot(df):
    """Create plot showing train and eval loss"""
    if df is None:
        return go.Figure()
    
    try:
        loss_data = (df
            .select([
                "iteration",
                "train_loss",
                "eval_loss"
            ])
            .sort("iteration"))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=loss_data.select("iteration").to_series().to_list(),
            y=loss_data.select("train_loss").to_series().to_list(),
            mode="lines+markers",
            name="Training Loss",
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=loss_data.select("iteration").to_series().to_list(),
            y=loss_data.select("eval_loss").to_series().to_list(),
            mode="lines+markers",
            name="Evaluation Loss",
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Model progress over all sites",
            xaxis_title="Iteration",
            yaxis_title="Average loss (MSE)",
            legend_title="Loss Type",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating loss plot: {str(e)}")
        return go.Figure()


def load_latest_traffic_data(accumulated_data=None) -> pl.DataFrame:
    """Load all available traffic prediction data"""
    try:
        pattern = "../data/model_output/traffic_data_to_model_*_prediction.parquet"
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No prediction files found matching pattern: {pattern}")
            return None
            
        all_data = []
        for file in files:
            try:
                df = pl.read_parquet(file)
                df = df.with_columns([
                    pl.col('measurement_time').str.strptime(
                        pl.Datetime,
                        format="%Y-%m-%d %H:%M:%S"
                    ).alias('measurement_time')
                ])
                
                df = df.with_columns([
                    pl.when(
                        pl.col('forecast_type') == "actual"
                    ).then(
                        pl.lit("Actual")
                    ).otherwise(
                        pl.lit("Prediction")
                    ).alias("type")
                ])
                
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                continue
        if not all_data:
            return None
        
        combined_df = pl.concat(all_data)
        combined_df = combined_df.unique(subset=['measurement_time', 'site_id', 'forecast_type'], keep='last')
        combined_df = combined_df.sort(['site_id', 'measurement_time'])
        if accumulated_data is not None:
            combined_df = pl.concat([accumulated_data, combined_df])
            combined_df = combined_df.unique(subset=['measurement_time', 'site_id', 'forecast_type'], keep='last')
            combined_df = combined_df.sort(['site_id', 'measurement_time'])
        
        logger.info(f"Successfully loaded and processed data with shape: {combined_df.shape}")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading traffic data: {str(e)}")
        return None
    
def load_loss_data():
    """Load the loss data"""
    try:
        loss_file = "../data/model_output/loss_data.parquet"
        if not os.path.exists(loss_file):
            logger.warning("Loss data file not found")
            return None
        df = pl.read_parquet(loss_file)        

        df = df.with_columns([
            pl.col("measurement_time").rank().alias("iteration")
        ])        
        return df
    except Exception as e:
        logger.error(f"Error loading loss data: {str(e)}")
        return None

# ------ #
#  MAIN  #
# ------ #

def main():
    st.title("Real-time traffic flow")
    st.markdown("Real-time traffic flow monitoring and prediction system")
    st.markdown("Alexander Gutell (alex.gutell@gmail.com) and Dan Vicente (dan.vicente.ihanus@gmail.com)")

    current_site_id = st.radio(
        "Select Site",
        list(d.keys()),
        format_func=lambda x: f"Site {x}",
        horizontal=True
    )
    
    site_plots = {}
    site_plots[current_site_id] = st.empty()
    
    metrics_container = st.container()
    with metrics_container:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Training Progress")
            loss_plot_container = st.empty()
        with col2:
            st.subheader(f"Traffic Flow Map - Site {current_site_id}")
            map_container = st.empty()
    
    status_container = st.container()
    with status_container:
        status_placeholder = st.empty()
        auto_refresh = st.checkbox('Enable Auto-refresh (5s)', value=True)
        last_refresh = st.empty()
    
    last_processed_file = ""
    accumulated_data = None
    
    while True:
        try:
            files = glob.glob("../data/model_output/traffic_data_to_model_*_prediction.parquet")
            
            if files:
                def get_iteration_number(filepath):
                    try:
                        filename = os.path.basename(filepath)
                        number_part = filename.split('model_')[-1].split('_prediction')[0]
                        return int(number_part)
                    except (IndexError, ValueError):
                        return -1
                
                current_file = max(files, key=get_iteration_number)
                
                if current_file != last_processed_file:
                    logger.info(f"Processing new file: {current_file}")
                    accumulated_data = load_latest_traffic_data(accumulated_data)
                    loss_data = load_loss_data()
                    current_time = int(time.time())
                    
                    if accumulated_data is not None:
                        status_placeholder.success("New data loaded successfully!")
                        last_processed_file = current_file
                        
                        traffic_fig = create_traffic_plot(accumulated_data, current_site_id)
                        site_plots[current_site_id].plotly_chart(
                            traffic_fig, 
                            use_container_width=True,
                            key=f"traffic_plot_{current_site_id}_{current_time}"
                        )
                        
                        if loss_data is not None:
                            loss_fig = create_loss_plot(loss_data)
                            loss_plot_container.plotly_chart(
                                loss_fig, 
                                use_container_width=True,
                                key=f"loss_plot_{current_time}"
                            )
                        
                        with col2:
                            map_obj = create_map(current_site_id)
                            map_container.empty()
                            map_container = st_folium(
                                map_obj,
                                width=600,
                                height=400,
                                key=f"map_{current_site_id}_{current_time}",
                                returned_objects=[]
                            )
                    else:
                        status_placeholder.warning("Could not load data from new file")
                else:
                    status_placeholder.info("Waiting for new data...")
            else:
                status_placeholder.warning("No prediction files found")
            
            last_refresh.text(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if auto_refresh:
                time.sleep(5)
            else:
                break
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            status_placeholder.error(f"An error occurred: {str(e)}")
            time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            status_placeholder.error(f"An error occurred: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    main()