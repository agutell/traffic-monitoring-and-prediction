import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
import os
import math
import time

# --- set CPU usage --- # 
available_cpu = os.cpu_count()
num_threads = math.floor(0.7 * available_cpu)
torch.set_num_threads(num_threads)
# ------––--------------- #

class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, steps, num_layers=1) -> None:
        super(TrafficGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.steps = steps
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def autoregress(self, pred, h_prev, input):
        pred_list=[]
        pred_list.append(pred)
        prev_pred = pred.unsqueeze(1)
        for i in range(self.steps-1):
            input = input[:, 1:, :]
            input = torch.cat((input, prev_pred), dim=1)
            out_seq, h = self.gru(input, h_prev)
            out_last = out_seq[:, -1, :] 
            pred = self.fc(out_last)
            pred_list.append(pred)
            prev_pred = pred.unsqueeze(1) 
            h_prev = h
        return pred_list
    
    def forward(self, input):
        input = input 
        input = input.float() 
        h_init = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        h_init = nn.init.xavier_uniform_(h_init) 
        out_seq, h_last = self.gru(input, h_init)
        out_last = out_seq[:, -1, :]
        first_pred = self.fc(out_last)
        pred_list = self.autoregress(first_pred, h_last, input)
        return pred_list
    
    def predict(self, model, input):
        model.eval()
        with torch.no_grad():
            predictions = model(input)
        return predictions

def train(epochs, model, optimizer, criterion, train_loader, eval_loader):
    early_stop = False
    prev_running_eval_loss = torch.inf
    train_loss = torch.inf
    eval_loss = torch.inf
    patience_counter = 0
    
    for epoch in range(epochs):
        if early_stop:
            break
        
        running_train_loss = 0
        batch_count = 0
        model.train()
        for data in train_loader:
            input, labels = data
            optimizer.zero_grad()
            prediction = model(input)
            prediction = torch.stack(prediction, dim=0).permute(1, 0, 2) #adjust dim
            loss = criterion(prediction, labels.float())
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_count += 1

        if batch_count != 0: 
            train_loss = running_train_loss / (batch_count * 2 * 5 * 3) # NOTE: Hard coded atm. batchsize=2, steps=5, num_stations=3
        if epoch % 20 == 0:
            if batch_count == 0: # NOTE: Dont need this since batch_count is never 0
                print('train_loss: ', running_train_loss) 
            else:
                print('train_loss: ', train_loss)

        running_eval_loss = 0
        batch_count = 0
        model.eval()
        with torch.no_grad():
            for data in eval_loader:
                input, labels = data
                prediction = model(input)
                prediction = torch.stack(prediction, dim=0).permute(1, 0, 2) #adjust dim
                running_eval_loss += criterion(prediction, labels).item()
                batch_count += 1
            if batch_count != 0:
                eval_loss = running_eval_loss/ (batch_count * 2 * 5 * 3) # NOTE: Hard coded atm. batchsize=2, steps=5, num_stations=3
            if epoch % 20 == 0:
                if batch_count == 0: # NOTE: Dont need this since batch_count is never 0
                    print('eval loss: ', running_eval_loss)
                else: 
                    print('eval loss: ', eval_loss)
            
        # Patience
        if running_eval_loss > prev_running_eval_loss:
            patience_counter += 1
            if patience_counter == 5:
                early_stop = True
                break
        else: 
            patience_counter = 0

        prev_running_eval_loss = running_eval_loss
    return train_loss, eval_loss


def load_df_to_tensor(data_path, station_ids, time_window):
    """
    Load a dataframe stored as parquet and convert it to a tensor.

    Args:
        data_path (string): File path to data
        station_ids (list): 

    Returns:
        tensor: Columns w.r.t. the most recent (time_window) measurement times.
                Rows w.r.t. site id.
                Vehicle flow rates as values
                
        df_pivot: Dataframe with the same structure as tensor, but with all measurement times.
    """
    df_original = pd.read_parquet(data_path, engine='pyarrow')
    filtered_df = df_original[df_original['site_id'].isin(station_ids)]
    filtered_df = filtered_df.sort_values(by=['site_id', 'measurement_time'], ascending=[True, False])
    
    recent_flow_rates = filtered_df.groupby('site_id', group_keys=False).apply(lambda x: x.head(time_window))
    recent_flow_rates = recent_flow_rates[['site_id', 'measurement_time', 'vehicle_flow_rate']]

    df_pivot = recent_flow_rates.pivot(index='site_id', columns='measurement_time', values='vehicle_flow_rate')
    df_pivot = df_pivot.sort_index(axis=1)

    num_nans_before = df_pivot.isna().sum().sum()
    
    df_pivot = df_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
    df_pivot = df_pivot.fillna(0)
    
    num_nans_after = df_pivot.isna().sum().sum()
    num_nans_filled = num_nans_before - num_nans_after
    print(f"Number of NaN values filled: {num_nans_filled}")
    print(df_pivot.head())

    tensor = torch.tensor(df_pivot.values)

    return tensor, df_pivot


def moving_average_forecast(df_original, station_ids, window_size=5, steps_ahead=5):
    """
    Generate forecasts using an autoregressive moving average of historical data.
    
    Args:
        df_original (DataFrame): Original traffic data
        station_ids (list): List of station IDs to forecast
        window_size (int): Number of past measurements to average
        steps_ahead (int): Number of steps to forecast
        
    Returns:
        DataFrame: Forecasted values in same format as df_gru_melted
    """
    # Filter data for selected stations
    filtered_df = df_original[df_original['site_id'].isin(station_ids)]
    # Sort values with measurement_time in descending order (newest first)
    filtered_df = filtered_df.sort_values(by=['site_id', 'measurement_time'], ascending=[True, False])
    
    # Calculate moving average for each station
    forecasts = []
    
    for site_id in station_ids:
        # Get data for this site
        site_data = filtered_df[filtered_df['site_id'] == site_id]
        
        # Get the most recent data points (already sorted with newest first)
        recent_data = site_data.head(window_size).copy()
        
        # Get the most recent timestamp
        last_timestamp = pd.to_datetime(recent_data['measurement_time'].iloc[0])
        
        # Store recent values in a list (reversed to put oldest first)
        recent_values = recent_data['vehicle_flow_rate'].values.tolist()
        recent_values.reverse()  # Now in chronological order (oldest first)
        
        # Generate forecasts autoregressively
        current_timestamp = last_timestamp
        for i in range(steps_ahead):
            # Calculate moving average of the last window_size values
            moving_avg = sum(recent_values[-window_size:]) / window_size
            
            # Advance timestamp
            current_timestamp = current_timestamp + pd.Timedelta(minutes=1)
            timestamp_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add forecast to results
            forecasts.append({
                'site_id': site_id,
                'measurement_time': timestamp_str,
                'vehicle_flow_rate': moving_avg,
                'forecast_type': 'moving_avg'
            })
            
            # Add this prediction to recent values for next iteration
            recent_values.append(moving_avg)
    
    return pd.DataFrame(forecasts)


def generate_windows(data, window_size=5, label_size=1):
    """
    Create data points (inputs and labels) for GRU training.

    Args: 
        data (2D tensor): 
            Columns wrt the most recent (time_window) measurement times.
            Rows wrt site id.
            Values wrt vehicle flow rate.
        window_size (int): input sequence length to GRU.
        label_size (int): label sequence length to GRU.
    
    Returns: 
        inputs (3D tensor): (num batches, window_size, num stations) 
        labels (3D tensor): (num batches, label_size, num stations) 
    """
    len_sequence = data.shape[1]
    num_windows = len_sequence - (window_size + label_size) + 1 

    inputs = []
    labels = []

    for i in range(num_windows):
        input_window = data[:, i:i + window_size] 
        input_window = input_window.unsqueeze(0)  
        input_window = input_window.permute(0, 2, 1)  

        label_window = data[:, i + window_size:i + window_size + label_size]  
        label_window = label_window.unsqueeze(0)  
        label_window = label_window.permute(0, 2, 1)  

        inputs.append(input_window)
        labels.append(label_window)

    inputs = torch.cat(inputs)
    labels = torch.cat(labels)

    return inputs, labels


def create_data_loaders(input, labels, partition_percent=0.80):
    """
    Convert input and labels to Dataset. 
    Splits data into training/evaluation sets and load DataLoaders

    Args: 
        inputs (3D tensor): (num batches, input seq.len, num stations) 
        labels (3D tensor): (num batches, label seq.len, num stations)
        partion_percent (float): training data ratio

    Returns:
        train_loader (DataLoader): training data
        eval_loader (DataLoader): evaluation data
    """
    dataset = TensorDataset(input, labels)
    split_index = int(len(dataset) * partition_percent)
    train_indices = list(range(0, split_index))
    eval_indices = list(range(split_index, len(dataset)))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, eval_indices)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
    eval_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)
    return train_loader, eval_loader


if __name__ == "__main__":
    station_ids = [4472, 4195, 1051]
    flag = True
    loss_data = []
    count = 1 
    
    # Add a forecast method flag
    forecast_method = "both"  # Options: "gru", "moving_avg", "both"
    moving_avg_window = 5  # Window size for moving average calculation

    ##### GRU Related #####
    initial_training = True # Must be set to True if running GRU
    seq_len = 5
    label_size = 1
    num_steps_predict = 5

    ##### Data Related #####
    first_interval = 20  # minutes before starting to train
    time_window = first_interval  # total sequence length to train on
    previous_time = 0

    while flag:
        file_path = f"../data/traffic_data"
        try:
            if initial_training == True:
                print(f"waiting {first_interval} minutes to train and predict...")
                time.sleep(first_interval*60)

            ##### Load Data #####
            data, df_pivot = load_df_to_tensor(file_path, station_ids, time_window)
            df_original = pd.read_parquet(file_path, engine='pyarrow')  # Store for moving average
            
            while previous_time == df_pivot.columns[-1]:  # If no new data has arrived, wait and load again.
                time.sleep(60)
                data, df_pivot = load_df_to_tensor(file_path, station_ids, time_window)
                df_original = pd.read_parquet(file_path, engine='pyarrow')
                
            num_stations = data.shape[0]
            
            ##### GRU Model Forecasting #####
            if forecast_method in ["gru", "both"]:
                input, labels = generate_windows(data, seq_len, label_size)
                train_loader, eval_loader = create_data_loaders(input, labels, partition_percent=0.5)

                ##### Initialize Model #####
                if initial_training == True:
                    model = TrafficGRU(input_size=num_stations, hidden_size=1000, \
                                output_size=num_stations, steps=label_size, num_layers=1)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    initial_training = False

                ##### Train Model #####
                model.steps = label_size  # sets num autoreg steps in training
                train_loss, eval_loss = train(epochs=100, model=model, optimizer=optimizer, criterion=criterion, \
                    train_loader=train_loader, eval_loader=eval_loader)

                ##### Write Loss to Disc #####
                loss_data.append([train_loss, eval_loss, df_pivot.columns[-1]])
                df_loss = pd.DataFrame(loss_data, columns=['train_loss', 'eval_loss', 'measurement_time'])
                print(df_loss.head(count))
                df_loss.to_parquet("../data/model_output/loss_data.parquet", index=False)

                ##### Make Autoregressive Prediction #####
                model.steps = num_steps_predict  # sets num autoreg steps in prediction
                predictions = model.predict(model, data[:, -seq_len:].T.unsqueeze(0))
                predictions = torch.cat(predictions, dim=0).T

                ##### Convert GRU predictions to DataFrame #####
                last_timestamp = pd.to_datetime(df_pivot.columns[-1])
                new_column_names = [(last_timestamp + pd.Timedelta(minutes=(i + 1))).strftime('%Y-%m-%d %H:%M:%S') for i in range(num_steps_predict)]
                df_gru = pd.DataFrame(predictions.numpy(), columns=new_column_names)
                df_gru.insert(0, 'site_id', df_pivot.index.values)
                
                # Add forecast type
                df_gru_melted = df_gru.melt(id_vars=['site_id'], var_name='measurement_time', value_name='vehicle_flow_rate')
                df_gru_melted['forecast_type'] = 'gru'
                df_gru_melted['measurement_time'] = df_gru_melted['measurement_time'].astype(str)
            
            ##### Moving Average Forecasting #####
            if forecast_method in ["moving_avg", "both"]:
                df_ma = moving_average_forecast(df_original, station_ids, 
                                               window_size=moving_avg_window, 
                                               steps_ahead=num_steps_predict)
            
            ##### Combine results based on forecast method #####
            if forecast_method == "gru":
                df_melted = df_gru_melted
            elif forecast_method == "moving_avg":
                initial_training = False # GRU related
                df_melted = df_ma
            else:  # "both"
                df_melted = pd.concat([df_gru_melted, df_ma])
            
            ##### Add current data for comparison #####
            df_current = df_pivot.iloc[:, -num_steps_predict:].reset_index()
            df_current_melted = df_current.melt(id_vars=['site_id'], var_name='measurement_time', value_name='vehicle_flow_rate')
            df_current_melted['forecast_type'] = 'actual'
            df_current_melted['measurement_time'] = df_current_melted['measurement_time'].astype(str)
            
            ##### Combine all data #####
            df_combined = pd.concat([df_melted, df_current_melted])
            df_combined = df_combined.sort_values(by=['site_id', 'measurement_time', 'forecast_type']).reset_index(drop=True)

            print(df_combined.head(30))
            df_combined.to_parquet(f"../data/model_output/traffic_data_to_model_{count}_prediction.parquet", index=False)
            
            count += 1
            previous_time = df_pivot.columns[-1]
            
        except FileNotFoundError:
            time.sleep(10)
            pass