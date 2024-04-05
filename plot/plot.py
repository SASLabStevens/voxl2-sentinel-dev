import os
import subprocess
import glob
import pandas as pd
import folium

def adb_pull(source, destination):
    # Executing the ADB pull command
    try:
        subprocess.run(['adb', 'pull', source, destination], check=True)
        print(f"Data successfully pulled from {source} to {destination}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pulling data: {e}")

def find_max_int_folder(base_path):
    # Find all the folders in the base path
    folders = glob.glob(os.path.join(base_path, '*'))
    # Filter out and keep only those folders that can be converted to integers
    int_folders = [f for f in folders if os.path.basename(f).isdigit()]
    # Find the folder with the maximum integer value
    max_int_folder = max(int_folders, key=lambda x: int(os.path.basename(x)), default=None)
    return max_int_folder

def save_map_in_folder(map_object, folder_name='gps_map'):
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Find the next available file name
    index = 0
    while True:
        file_path = os.path.join(folder_name, f'gps_map_{index}.html')
        if not os.path.exists(file_path):
            break
        index += 1

    # Save the map
    map_object.save(file_path)
    print(f"GPS map has been saved as '{file_path}'")

def plot_gps_data(csv_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    filtered_data = data.dropna(subset=['latitude', 'longitude'])
    # Filter out rows where latitude and longitude are both 0
    filtered_data = data[(data['latitude'] != 0) & (data['longitude'] != 0)]
    
    if filtered_data.empty:
        print("No valid GPS data available.")
        return None
    
    # Create a folium map centered around the mean coordinates
    m = folium.Map(location=[filtered_data['latitude'].mean(), filtered_data['longitude'].mean()], zoom_start=15)

    # Add data points to the map
    for _, row in filtered_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=1,  # Adjust as needed
            color='blue',
            fill=True,
            fill_color='blue',
        ).add_to(m)

    return m

def plot_odo_data(csv_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Filter out rows where latitude and longitude are both 0
    filtered_data = data[(data['odo_x'] != 0) & (data['odo_y'] != 0)]

    # Create a line plot
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['odo_x'], filtered_data['odo_y'], marker='o', color='b')
    plt.title("Odo Line Plot")
    plt.xlabel("odo_x (Latitude)")
    plt.ylabel("odo_y (Longitude)")
    plt.grid(True)
    plt.show()

# Path where the ADB pulled data will be stored
if __name__ == "__main__":
    destination_path = './'

    # ADB pull command (modify the source path as per your need)
    adb_pull('/metadata', destination_path)

    # Base path to your folders (assuming the data is pulled into this directory)
    base_path = './metadata/StevensLibrary/GOPROMINI11'

    # Find the folder with the maximum integer value
    max_int_folder = find_max_int_folder(base_path)

    if max_int_folder:
        csv_file_path = os.path.join(max_int_folder, 'info.csv')
        # Call the function to plot the data
        map_obj = plot_gps_data(csv_file_path)
        if map_obj is not None:
            save_map_in_folder(map_obj)
        plot_odo_data(csv_file_path)
    else:
        print("No suitable folder found.")
    
