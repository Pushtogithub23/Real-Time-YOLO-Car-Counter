import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv

# Initialize YOLO model and video info
model = YOLO("yolo11x.pt")  # Replace with your model path
video_path = "DATA/INPUTS/cars_on_highway_5.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# Setup tracker, smoother, and class filters
tracker = sv.ByteTrack(frame_rate=fps)
smoother = sv.DetectionsSmoother()
vehicle_classes = {'car', 'motorbike', 'bus', 'truck'}
selected_classes = [cls_id for cls_id, class_name in model.names.items() if class_name in vehicle_classes]

# Define counting zones (scaled)
arr1 = np.array([[761, 642], [1073, 642], [1070, 732], [968, 776], [872, 1038], [97, 1049]], dtype=np.int32)
arr2 = np.array([[1105, 639], [1402, 645], [1920, 959], [1920, 1080], [930, 1073], [991, 811], [1105, 755]],
                dtype=np.int32)
zone_points = [np.floor(arr * 0.66).astype(np.int32) for arr in [arr1, arr2]]
zones = [sv.PolygonZone(points) for points in zone_points]
colors = sv.ColorPalette.from_hex(['#ef260e', '#07f921'])  # Colors for zones

# Initialize count storage and ID tracking
total_counts, crossed_ids = [], set()
counts_up, ids_up = [], set()
counts_down, ids_down = [], set()

# Display settings
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)


# Overlay helper function
def draw_overlay(frame, points, color, alpha=0.25):
    overlay = frame.copy()
    cv.fillPoly(overlay, [points], color)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# Counting functions for zones
def count_vehicle_in_zone(ID, cx, cy, zone_idx):
    if cv.pointPolygonTest(zone_points[zone_idx], (cx, cy), False) >= 0:
        if ID not in crossed_ids:
            total_counts.append(ID)
            crossed_ids.add(ID)
        if zone_idx == 0 and ID not in ids_down:
            counts_down.append(ID)
            ids_down.add(ID)
        elif zone_idx == 1 and ID not in ids_up:
            counts_up.append(ID)
            ids_up.add(ID)


# Annotation function for zones and detections
def annotate_frame(frame, detections):
    detections = detections[np.isin(detections.class_id, selected_classes)]

    for points, color in zip(zone_points, [(88, 117, 234), (11, 244, 113)]):
        draw_overlay(frame, points, color=color, alpha=0.25)

    for idx, zone in enumerate(zones):
        zone_annotator = sv.PolygonZoneAnnotator(zone, thickness=4, color=colors.by_idx(idx), text_scale=2,
                                                 text_thickness=2)
        mask = zone.trigger(detections)
        filtered_detections = detections[mask]

        # Draw boxes and labels for filtered detections
        box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        label_annotator = sv.LabelAnnotator(text_scale=text_scale,
                                            text_thickness=thickness,
                                            text_position=sv.Position.TOP_CENTER,
                                            color_lookup=sv.ColorLookup.TRACK)

        box_annotator.annotate(frame, filtered_detections)
        label_annotator.annotate(
            frame, detections=filtered_detections,
            labels=[f"{model.names[class_id]} #{trk_id}" for class_id, trk_id in
                    zip(filtered_detections.class_id,
                        filtered_detections.tracker_id)]
        )
        zone_annotator.annotate(frame)

    # Count vehicles based on their bottom center coordinates
    for track_id, bottom_center in zip(detections.tracker_id,
                                       detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
        cx, cy = map(int, bottom_center)
        cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)
        count_vehicle_in_zone(track_id, cx, cy, 0)
        count_vehicle_in_zone(track_id, cx, cy, 1)

    # Display counters on the frame
    counter_labels = [f"COUNTS: {len(total_counts)}", f"UP: {len(counts_up)}", f"DOWN: {len(counts_down)}"]
    count_colors = [(0, 0, 0), (6, 104, 2), (0, 0, 255)]
    cv.rectangle(frame, (0, 0), (300, 150), (255, 255, 255), cv.FILLED)
    for i, (label, color) in enumerate(zip(counter_labels, count_colors)):
        cv.putText(frame, label, (20, 50 + i * 40), cv.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)


# Process video
cap = cv.VideoCapture(video_path)
output_path = "DATA/OUTPUTS/car_counter_yolo11x.mp4"
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting and tracking the model with YOLO
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    if detections.tracker_id is not None:
        annotate_frame(frame, detections)

    # Writing the frames to save the video
    out.write(frame)
    cv.imshow("Video", frame)

    # Close the window by pressing 'p'
    if cv.waitKey(1) & 0xff == ord('p'):
        break

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()
