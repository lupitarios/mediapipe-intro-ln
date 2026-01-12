import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing_utils
import mediapipe.python.solutions.drawing_styles as drawing_styles

# Initialize MediaPipe Hands.
hands = mp_hands.Hands(
    static_image_mode=False, #Set to False for video input
    max_num_hands=2,        #Detect up to 2 hands
    min_detection_confidence=0.5, #Minimum confidence for detection
)

# Start capturing video from the webcam.
cap = cv.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the webcam.
    success, frame = cap.read()

    # If the frame was not read successfully, break the loop.
    if not success:
        print("Camera not found.")
        continue

    # Convert the BGR image to RGB.
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the RGB image to detect hands and tracking.
    hands_detected = hands.process(frame)

    # Convert the image back to BGR for rendering.
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # If hands are detected, draw the landmarks and connections.
    if hands_detected.multi_hand_landmarks:
        for hands_landmarks in hands_detected.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hands_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

    # Display the frame with hand landmarks.
    cv.imshow('MediaPipe Hands', frame)

    # Break the loop if the 'Esc' key is pressed.
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows.
cap.release()
cv.destroyAllWindows()