# Copyright 2025 xAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import pyautogui
import speech_recognition as sr
from cv2 import __version__

# Print OpenCV version for attribution
print(f"Using OpenCV {__version__}")

# Initialize speech recognizer and adjust for ambient noise
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)

# Function to listen for "go" command
def listen():
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=5)
            return r.recognize_google(audio).strip().lower()
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            return ""

# Initialize webcam and previous coordinates
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Find contours (pupil)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cx, cy = int(x + w/2), int(y + h/2)

        # Smooth prediction
        if abs(cx - prev_x) < 10:
            pred_x = cx + (cx - prev_x)
            pred_y = cy + (cy - prev_y)
        else:
            pred_x, pred_y = cx, cy
        prev_x, prev_y = cx, cy

        # Move cursor to predicted position
        pyautogui.moveTo(pred_x * 3.5, pred_y * 3.5, duration=0.05)

        # Check for "go" command to click
        if listen() == "go":
            pyautogui.click()

        # Draw ghost cursor
        cv2.circle(frame, (pred_x, pred_y), 10, (255, 0, 0), 2)

    # Display frame
    cv2.imshow('Ghost Cursor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
