import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 Model
def load_model(weights_path):
    return YOLO(weights_path)  # Load custom trained model

# Process video and count total vehicles
def process_video(model, input_video_path, output_video_path, excel_path):
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video details
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dictionary to store total vehicle counts
        vehicle_counts = {}
        # Perform YOLO inference on the frame
        results = model.predict(frame, imgsz=224, conf=0.25)

        detected_vehicles = set()  # To store unique detections per frame

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            classes = result.boxes.cls.cpu().numpy()  # Class indices
            
            for i, box in enumerate(boxes):
                label = model.names[int(classes[i])]  # Get class name
                
                # Only count each detected vehicle once per frame
                if label not in detected_vehicles:
                    detected_vehicles.add(label)
                    
                    # Increment the total count of detected vehicles
                    if label in vehicle_counts:
                        vehicle_counts[label] += 1
                    else:
                        vehicle_counts[label] = 1

                # Draw bounding box and label on frame
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)  # Green color
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame = cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        

        # Write processed frame to output video
        out.write(frame)
        
        # Show frame (optional)
        cv2.imshow('Vehicle Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Convert vehicle count dictionary to DataFrame and save as Excel file
    df = pd.DataFrame(list(vehicle_counts.items()), columns=['Vehicle Type', 'Total Count'])
    df.to_excel(excel_path, index=False)

    print(f"Processing complete! Output video saved as {output_video_path}")
    print(f"Total vehicle count data saved to {excel_path}")

if __name__ == "__main__":
    # Path to your trained YOLOv8 model weights
    model_weights = "nevil.pt"
    
    # Path to input video, output video, and Excel file
    input_video = "test.mp4"
    output_video = "output_traffic_video.mp4"
    excel_output = "vehicle_count.xlsx"
    
    # Load model
    model = load_model(model_weights)
    
    # Process video and count total vehicles
    process_video(model, input_video, output_video, excel_output)
