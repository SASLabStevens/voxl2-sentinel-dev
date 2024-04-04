import pandas as pd
import numpy as np
import argparse

def calculate_gps_coordinates(df, start_latitude, start_longitude, start_altitude):
    # Initialize the lists to store the calculated GPS coordinates
    calculated_latitudes = [start_latitude]
    calculated_longitudes = [start_longitude]
    calculated_altitudes = [start_altitude]

    # Loop through the odometry data and update the GPS coordinates
    for index, row in df.iterrows():
        # Calculate the displacement
        delta_x = row['odo_x']  # Assuming odo_x is in meters
        delta_y = row['odo_y']  # Assuming odo_y is in meters
        delta_z = row['odo_z']  # Assuming odo_z is in meters

        # Convert the displacement to changes in latitude, longitude, and altitude
        delta_latitude = delta_y / 111111  # in degrees
        delta_longitude = delta_x / (111111 * np.cos(np.radians(start_latitude)))  # in degrees

        # Update the GPS coordinates
        new_latitude = calculated_latitudes[-1] + delta_latitude
        new_longitude = calculated_longitudes[-1] + delta_longitude
        new_altitude = start_altitude - delta_z
        
        # Append the new coordinates to the lists
        calculated_latitudes.append(new_latitude)
        calculated_longitudes.append(new_longitude)
        calculated_altitudes.append(new_altitude)

    # Add the calculated coordinates to the dataframe
    df['calculated_latitude'] = calculated_latitudes[1:]
    df['calculated_longitude'] = calculated_longitudes[1:]
    df['calculated_altitude'] = calculated_altitudes[1:]


    return df

def main(args):
    # Load the data
    df = pd.read_csv(args.input_file)

    # Calculate the GPS coordinates
    df = calculate_gps_coordinates(df, args.start_latitude, args.start_longitude, args.start_altitude)

    # Save the updated dataframe to a CSV file
    df.to_csv(args.output_file, index=False)

    # Display the first few rows of the updated dataframe
    print(df[['datetime', 'latitude', 'longitude', 'altitude', 'calculated_latitude', 'calculated_longitude', 'calculated_altitude']].head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate GPS coordinates from odometry data.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing the odometry data.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file where the updated data will be saved.')
    parser.add_argument('--start_latitude', type=float, required=True, help='Starting latitude.')
    parser.add_argument('--start_longitude', type=float, required=True, help='Starting longitude.')
    parser.add_argument('--start_altitude', type=float, required=True, help='Starting altitude.')

    args = parser.parse_args()
    main(args)


'''
python odo2gps.py --input_file '/home/gaudel/Desktop/vddc/data/3/info.csv' --start_latitude 40.745058 --start_longitude -74.024718 --start_altitude 22.075342 --output_file '/home/gaudel/Desktop/stevensCollection3/metadata/3/modified.csv' 

'''
