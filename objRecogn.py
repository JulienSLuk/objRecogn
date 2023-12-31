import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load classes
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera or replace with your camera index

# Create a Matplotlib figure
plt.figure()

frame_number = 0  # Track frame number

def get_average_color(image, x, y, w, h):
    roi = image[y:y + h, x:x + w]  # Extract the region of interest
    average_color = np.mean(roi, axis=(0, 1))  # Calculate the average color in the region
    return tuple(average_color.astype(int))

while True:
    ret, frame = cap.read()
    
    frame_number += 1  # Increment frame number
    
    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform forward pass
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    # Process the results
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            
            # Check if the scores array has a valid length
            if len(scores) > 0:
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust this threshold as needed
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
    
    # Non-maximum suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and labels with class name, confidence, and color
    for i in range(len(boxes)):
        if i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = get_average_color(frame, x, y, w, h)  # Get the average color
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            
            # Additional details: position, size, and color
            text += f" | Position: ({x}, {y})"
            text += f" | Size: ({w}, {h})"
            text += f" | Color: {color}"
            
            cv2.putText(frame, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Overlay frame number and timestamp
    timestamp = cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame with detections using Matplotlib
    plt.clf()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.pause(0.01)  # Add a small pause to update the figure
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
