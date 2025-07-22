import numpy as np
import cv2
import imutils
import time

gun_cascade = cv2.CascadeClassifier('gun_cascade.xml')
if gun_cascade.empty():
    print("❌ Failed to load gun_cascade.xml")
    exit()

camera = cv2.VideoCapture(0)
gun_detected_logged = False
gun_exists = False

print("[INFO] Warming up camera...")
time.sleep(2)  # Give camera time to initialize

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try to reduce false positives
    gun = gun_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,      # Was 1.1
    minNeighbors=12,      # Was 8 → higher = stricter
    minSize=(150, 150)    # Was 100x100
)

    if len(gun) > 0:
        gun_exists = True
        if not gun_detected_logged:
            print("[INFO] Gun detected in frame!")
            gun_detected_logged = True  # Only print once

    for (x, y, w, h) in gun:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show detection count on screen
    cv2.putText(frame, f"Guns detected: {len(gun)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Security Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

if gun_exists:
    print("✅ Gun detected!")
else:
    print("❌ No gun detected.")
