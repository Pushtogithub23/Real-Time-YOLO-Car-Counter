import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# Initialize YOLO model and video info
model = YOLO("../yolo weights/yolov8n.pt")  # Replace with your model path
video_path = "DATA/INPUTS/cars_on_highway_1.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# Setup annotators
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)

label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                    text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps,
                                    position=sv.Position.CENTER, color_lookup=sv.ColorLookup.TRACK)

# Tracker and smoother setup
tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()

# Vehicle class setup
class_names = model.names
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']
selected_classes = [cls_id for cls_id, class_name in model.names.items() if
                    class_name in vehicle_classes]  # Get class IDs for vehicles

# Initialize counters
limits = [400, 400, 1250, 400]  # Line for vehicle counting
total_counts, crossed_ids = [], set()


def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
    """Draws a semi-transparent overlay rectangle."""
    overlay = frame.copy()
    rect_color = color if filled else (0, 0, 0)
    cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def count_vehicles(track_id, cx, cy, limits, crossed_ids):
    """Counts vehicles crossing the line."""
    if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10 and track_id not in crossed_ids:
        crossed_ids.add(track_id)
        return True
    return False


def draw_tracks_and_count(frame, detections, total_counts, limits):
    """Annotates the frame with detected tracks and counts vehicles."""
    # Filter by vehicle classes and detection confidence
    detections = detections[(np.isin(detections.class_id, selected_classes)) & (detections.confidence > 0.5)]

    # Annotating the Bounding Boxes, Labels and Traces
    labels = [f"#{track_id} {class_names[cls_id]}" for track_id, cls_id in
              zip(detections.tracker_id, detections.class_id)]

    box_annotator.annotate(frame, detections=detections)
    label_annotator.annotate(frame, detections=detections, labels=labels)
    trace_annotator.annotate(frame, detections=detections)

    # Annotate tracks and vehicles
    for track_id, center_point in zip(detections.tracker_id,
                                      detections.get_anchors_coordinates(anchor=sv.Position.CENTER)):
        cx, cy = map(int, center_point)

        cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)  # Draw vehicle center point

        if count_vehicles(track_id, cx, cy, limits, crossed_ids):
            total_counts.append(track_id)
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.ROBOFLOW, thickness=4)
            draw_overlay(frame, (400, 300), (1250, 500), alpha=0.25, color=(10, 255, 50))

    sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=110, y=30), sv.Color.ROBOFLOW, 1.25,
                 2, background_color=sv.Color.WHITE)


cap = cv.VideoCapture(video_path)
output_path = "DATA/OUTPUTS/car_counter_2.mp4"
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

# Video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define the Region of Interest for tracking
    crop = frame[225:, 220:]
    mask_b = np.zeros_like(frame, dtype=np.uint8)
    mask_w = np.ones_like(crop, dtype=np.uint8) * 255
    mask_b[225:, 220:] = mask_w

    # Apply the mask to the original frame
    ROI = cv.bitwise_and(frame, mask_b)

    # YOLO detection and tracking
    results = model(ROI)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    if detections.tracker_id is not None:
        # Draw counting line and process vehicle tracks
        sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]), end=sv.Point(x=limits[2], y=limits[3]),
                     color=sv.Color.RED, thickness=4)
        draw_overlay(frame, (400, 300), (1250, 500), alpha=0.2)
        draw_tracks_and_count(frame, detections, total_counts, limits)

    # Write to save the frames
    out.write(frame)
    # Display frame
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xff == ord('p'):  # Pause with 'p'
        break

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()
