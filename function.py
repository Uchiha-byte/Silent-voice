import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to perform MediaPipe hand landmark detection
def mediapipe_detection(image, model):
    # Convert color space from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Make prediction
    results = model.process(image_rgb)
    
    # Convert color space back from RGB to BGR
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image, results

# Function to draw hand landmarks on the image
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# Function to extract hand keypoints
def extract_keypoints(results):
    keypoints_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21 * 3)
            keypoints_list.append(rh)
    
    return np.concatenate(keypoints_list) if keypoints_list else np.zeros(21 * 3)

# Path for exported data (numpy arrays)
DATA_PATH = os.path.join('MP_Data') 

# List of hand gestures (actions)
actions = np.array(['Hello', 'I love you', 'No', 'Thanks', 'Yes'])

# Number of sequences and sequence length
no_sequences = 30
sequence_length = 30

# Create directories for data storage
for action in actions:
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Set up webcam
cap = cv2.VideoCapture(0)

# Set up MediaPipe hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Main loop for collecting data
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # Read frame from webcam
                ret, frame = cap.read()

                # Perform hand landmark detection
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks on the image
                draw_styled_landmarks(image, results)

                # Display information on the frame
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Show the frame
                cv2.imshow('OpenCV Feed', image)

                # Export keypoints to a NumPy file
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
