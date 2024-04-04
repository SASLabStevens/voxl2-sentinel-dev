import dv_processing as dv
import os
from datetime import timedelta
from PIL import Image

# Camera name and initial setup
camera_name = "DVXplorer_mini"  # Replace with your actual camera name

# Open any camera that is discovered in the system
camera = dv.io.CameraCapture()

# Check whether streams are available
eventsAvailable = camera.isEventStreamAvailable()
framesAvailable = camera.isFrameStreamAvailable()
imuAvailable = camera.isImuStreamAvailable()
triggersAvailable = camera.isTriggerStreamAvailable()

# Specify the root folder path where the data will be saved
root_save_folder = "./metadata/event"

# Function to create next integer-named folder
def create_next_integer_folder(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    subfolders = [f.name for f in os.scandir(base_path) if f.is_dir() and f.name.isdigit()]
    max_number = max([int(folder) for folder in subfolders] + [0])
    next_folder_name = f"{max_number + 1}"
    next_folder_path = os.path.join(base_path, next_folder_name)
    os.makedirs(next_folder_path, exist_ok=True)
    return next_folder_path

# Create the next integer-named folder
save_folder = create_next_integer_folder(root_save_folder)
output_file_path = os.path.join(save_folder, "eventdata.aedat4")

# Initialize visualizer with the resolution of the camera
visualizer = dv.visualization.EventVisualizer(camera.getEventResolution())

# Apply color scheme configuration to the visualizer
visualizer.setBackgroundColor(dv.visualization.colors.black())
visualizer.setPositiveColor(dv.visualization.colors.red())
visualizer.setNegativeColor(dv.visualization.colors.iniBlue())

# Initialize an accumulator with some resolution
accumulator = dv.Accumulator(camera.getEventResolution())

# Apply configuration to the accumulator
accumulator.setMinPotential(0.0)
accumulator.setMaxPotential(1.0)
accumulator.setNeutralPotential(0.5)
accumulator.setEventContribution(0.15)
accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
accumulator.setDecayParam(1e+6)
accumulator.setIgnorePolarity(False)
accumulator.setSynchronousDecay(False)

# Initialize a slicer
slicer = dv.EventStreamSlicer()

# Initialize a frame counter for unique filenames
frame_counter = 0

# Declare the callback method for slicer to save the images
def slicing_callback(events: dv.EventStore):
    global frame_counter
    # Visualize the events
    visualizer_image = visualizer.generateImage(events)
    # Convert to PIL image format to save
    pil_image = Image.fromarray(visualizer_image)
    # Generate a unique filename for each frame
    filename = os.path.join(save_folder, f"frame_{frame_counter:05d}.png")
    # Save the frame to disk
    pil_image.save(filename)
    frame_counter += 1

# Register the callback to save images every 2000 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

try:
    # Open a file to write, will allocate streams for all available data types
    writer = dv.io.MonoCameraWriter(output_file_path, camera)

    print("Start recording to", output_file_path)
    while camera.isConnected():
        if eventsAvailable:
            # Get Events
            events = camera.getNextEventBatch()
            # Write Events and pass them into the slicer
            if events is not None:
                writer.writeEvents(events, streamName='events')
                slicer.accept(events)

        if framesAvailable:
            # Get Frame and write it
            frame = camera.getNextFrame()
            if frame is not None:
                writer.writeFrame(frame, streamName='frames')

        if imuAvailable:
            # Get IMU data and write it
            imus = camera.getNextImuBatch()
            if imus is not None:
                writer.writeImuPacket(imus, streamName='imu')

        if triggersAvailable:
            # Get trigger data and write it
            triggers = camera.getNextTriggerBatch()
            if triggers is not None:
                writer.writeTriggerPacket(triggers, streamName='triggers')

except KeyboardInterrupt:
    print("Ending recording")

