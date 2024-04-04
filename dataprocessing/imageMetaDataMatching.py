import cv2
import pandas as pd
import json
import argparse
from datetime import datetime
import os

def get_frame_from_video(video_path, video_start_datetime, timestamp):
    """
    Extract a frame from the video at a specific timestamp.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Calculate the frame number based on the timestamp and video's FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_diff = datetime.strptime(timestamp, '%Y-%m-%d %H-%M-%S') - datetime.strptime(video_start_datetime, '%Y-%m-%d %H-%M-%S')
    frame_number = int(time_diff.total_seconds() * fps)

    # Set the video position to the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    success, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    return success, frame

def save_frame_and_data(frame, data, directory_timestamp, index, output_folder):
    """
    Save the frame as a PNG image and data as a JSON file with the specified structure in the output folder.
    """
    new_base_name = f"siteRTX0002-camA003-{directory_timestamp}-{str(index).zfill(8)}"
    image_name = f"{new_base_name}.png"
    json_name = f"{new_base_name}.json"

    # Construct file paths
    image_path = os.path.join(output_folder, image_name)
    json_path = os.path.join(output_folder, json_name)

    # Save the frame
    cv2.imwrite(image_path, frame)

    # Construct the JSON data with the specified structure
    json_data = {
        "version": "4.2.0",
        "fname": image_name,
        "site": "SIT_campus",
        "source": "camA003",
        "collection": directory_timestamp,
        "timestamp": datetime.strptime(data['datetime'], '%Y-%m-%d-%H-%M-%S').timestamp(),
        "extrinsics": {
            "lat": data['calculated_latitude'],
            "lon": data['calculated_longitude'],
            "alt": data['calculated_altitude'],
            "omega": data["camera_roll"], 
            "phi": data["camera_pitch"],
            "kappa": data["camera_yaw"]
        },
        "intrinsics": {
            "fx": 2810.0502403212413,
            "fy": 2810.368191925827,
            "cx": 2659.45774799179,
            "cy": 1506.4682591699664,
            "k1": 0.017490663559648095,
            "k2": -0.033434491338205564,
            "k3": 0.0239182967753969,
            "p1": 0.00024263145253182732,
            "p2": -0.0003124233062363525,
            "p3": 0,
            "p4": 0,
            "rows": 2988,
            "columns": 5312
        },
        "projection": "fisheye_pix4d",
        "calibration": "none",
        "geolocation": "gps",
        "pii_status": "n/a",
        "env_conditions": ["clear", "snow_on_ground"],
        "type": "airborne",
        "modes": [],
        "exterior": True,
        "interior": False,
        "transient_occlusions": [],
        "artifacts": [],
        "masks": {"key": {"rle": "", "method": ""}},
        "pii_detected": [],
        "pii_removed": []
    }

    # Save the JSON data
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def process_video_and_csv(video_path, csv_path, video_start_datetime, output_folder):
    """
    Process the video and CSV to extract frames and save data with the new naming convention.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert the video start datetime to a datetime object
    video_start_datetime_obj = datetime.strptime(video_start_datetime, '%Y-%m-%d %H:%M:%S')

    # Get the directory timestamp
    # directory_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the rows in the DataFrame
    for index, row in enumerate(df.iterrows(), start=1):
        row_datetime = datetime.strptime(row[1]['datetime'], '%Y-%m-%d %H:%M:%S')

        # Only process rows where the timestamp is greater than or equal to the video start time
        if row_datetime >= video_start_datetime_obj:
            success, frame = get_frame_from_video(video_path, video_start_datetime, row[1]['datetime'])

            if success:
                save_frame_and_data(frame, row[1].to_dict(), row[1]['datetime'], index, output_folder)
            else:
                print(f"Failed to extract frame for timestamp: {row[1]['datetime']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video based on CSV datetime.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    parser.add_argument("--start_datetime", required=True, help="Start datetime of the video (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder where images and JSON files will be saved.")
    args = parser.parse_args()

    process_video_and_csv(args.video, args.csv, args.start_datetime, args.output_folder)


'''
python imageMetaDataMatching.py --video '/home/gaudel/Desktop/stevensCollection3/flight1/GX020006.MP4' --csv '/home/gaudel/Desktop/stevensCollection3/metadata/1/modified.csv' --start_datetime "2024-02-19 07:16:09" --output_folder '/home/gaudel/Desktop/stevensCollection3/processed_data/1' 

'''
