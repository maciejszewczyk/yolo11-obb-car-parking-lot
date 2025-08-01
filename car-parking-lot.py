import cv2
from ultralytics import YOLO
import time
import argparse
import os
import numpy as np

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="best.pt", help="YOLO11 OBB model")  # Path to the YOLO model
parser.add_argument("--video", type=str, default="car-parking-lot.mp4", help="Path to input video or webcam index (0)")  # Input video file or webcam
parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold for detection")  # Confidence threshold for object detection
parser.add_argument("--save", action="store_true", help="Save the result")  # Option to save the output video
args = parser.parse_args()  # Parse command-line arguments

# Function to display FPS (Frames Per Second) on the frame
def show_fps(frame, fps):
    x, y, w, h = 10, 10, 350, 50  # Define the position and size of the FPS display area
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Draw a black rectangle for background
    cv2.putText(frame, "FPS: " + str(fps), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 255, 0), 3)  # Add FPS text to the frame

def show_information(frame, title, counter, box_color):    
    # Define the size of the box
    box_width = 250
    box_height = 50

    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Coordinates for the center of the frame
    center_x = frame_width // 2 
    center_y = 10  # The y-coordinate position for the top of the box    

    # Set starting and ending coordinates for the box (centered)
    start_x = center_x - box_width // 2
    start_y = center_y
    end_x = start_x + box_width
    end_y = start_y + box_height

    # Draw the box on the frame using cv2.rectangle
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, -1)  # Filled box

    # Create the text to display
    text = f"{title}: {counter}"
    if(counter == 0):
        text = f"{title}"
    
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_thickness = 3

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Coordinates to center the text inside the box
    text_x = start_x + (box_width - text_size[0]) // 2
    text_y = start_y + (box_height + text_size[1]) // 2

    # Draw the text inside the box using cv2.putText
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

def draw_box(frame, obb, class_id):
    # Extract the four corner points of the oriented bounding box (OBB)
    xy1 = obb[0]  # Top-left corner
    xy2 = obb[1]  # Top-right corner
    xy3 = obb[2]  # Bottom-right corner
    xy4 = obb[3]  # Bottom-left corner

    # Set the default box color to red (BGR format)
    box_color = (0, 0, 255)  
    # Change the box color to green if the class ID is 1
    if(class_id == 1):
        box_color = (0, 255, 0)

    # Create a NumPy array representing the four points of the OBB
    obb_points = np.array([xy1, xy2, xy3, xy4])            
    # Draw the oriented bounding box on the frame as a closed polyline
    cv2.polylines(frame, [obb_points], isClosed=True, color=box_color, thickness=2)

if __name__ == '__main__':
    # Set up video capture
    video_input = args.video
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)  # Open webcam if video_input is a digit
    else:
        cap = cv2.VideoCapture(video_input)  # Open video file

    # Save Video
    output_folder = "result"  # Directory to save the output video
    if(not os.path.isdir(output_folder)):  # Create the directory if it doesn't exist
        os.mkdir(output_folder)

    if args.save:  # If the save option is selected
        # Extract the filename from the input video and remove the extension
        filename = os.path.splitext(os.path.basename(args.video))[0]

        # Define the path for the output video
        output_video_path = f"{output_folder}/{filename}.mp4"  

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the input video        

        # Create video writer objects to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # Initialize the VideoWriter

    conf_thres = args.conf  # Confidence threshold for detection
    model = YOLO(args.model)  # Load YOLO11 model  

    start_time = 0  # Initialize start time for FPS calculation    

    while cap.isOpened():
        success, frame = cap.read()  # Read a frame from the video
        annotated_frame = frame

        if success:
            results = model(frame, conf=conf_thres, verbose=False)  

            # Extract the oriented bounding boxes (OBBs) and their class IDs from the detection results
            obbs = results[0].obb.xyxyxyxy.int().cpu().tolist()  # Get OBB coordinates as a list of integers
            class_ids = results[0].obb.cls.int().cpu().tolist()  # Get class IDs for the detected objects

            # Initialize a counter for empty parking lots
            empty_parking_lot = 0

            # Iterate through each OBB and its corresponding class ID
            for obb, class_id in zip(obbs, class_ids):
                # Check if the object represents an empty parking lot (class ID == 1)
                if(class_id == 1):
                    empty_parking_lot += 1  # Increment the counter for empty parking lots

                # Draw the oriented bounding box (OBB) on the annotated frame
                draw_box(annotated_frame, obb, class_id)                                         

            # Display parking lot status information on the frame
            box_color = (0, 255, 0)  # Default box color for "Empty" status (Green)
            box_title = "Empty"      # Default status title

            if(empty_parking_lot == 0):  # If no empty parking lots are detected
                box_color = (0, 0, 255)  # Change the box color to red
                box_title = "Full"       # Change the status title to "Full"                

            # Display the status and the number of empty parking lots on the frame
            show_information(annotated_frame, box_title, empty_parking_lot, box_color)
        
            # Calculate FPS
            end_time = time.time()  # Get the current time
            fps = 1 / (end_time - start_time)  # Calculate frames per second
            
            start_time = end_time  # Update start time for the next frame

            # Show FPS on the frame
            fps = float("{:.2f}".format(fps))  # Format FPS to two decimal places
            show_fps(annotated_frame, fps)  # Call function to display FPS

            resized_frame = cv2.resize(annotated_frame, (1280, 720)) 

            # Display the annotated frame in a fullscreen window
            cv2.namedWindow("YOLO11 Car Parking Lot Counter", cv2.WND_PROP_FULLSCREEN)  # Create a named window
            cv2.setWindowProperty("YOLO11 Car Parking Lot Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set the window to fullscreen            

            cv2.imshow("YOLO11 Car Parking Lot Counter", resized_frame)  # Show the annotated frame               

            if args.save:  # If the save option is selected
                writer.write(annotated_frame)  # Write the annotated frame to the output video file               

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Exit loop on 'q' key press
        
        else:
            # Break the loop if the end of the video is reached
            break

    if args.save:        
        print("The result will be saved in: "+ output_video_path)  # Print the save location of the output video

    # Release the video capture object and close the display window
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows