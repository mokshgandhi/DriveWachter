import cv2

def blur_sensitive_regions(frame, blur_model):
    results = blur_model(frame)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        roi = cv2.GaussianBlur(roi, (51, 51), 30)
        frame[y1:y2, x1:x2] = roi
    return frame
