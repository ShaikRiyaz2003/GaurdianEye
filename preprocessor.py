import cv2
import numpy as np
from keras.models import load_model
import pickle
import os
from tensorflow.keras.models import model_from_json, load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
def extract_and_process_frames(video_path):
    """
    Extracts every 10th frame from the given video file, converts each frame to grayscale,
    resizes it to 50x50 pixels, and returns them in a NumPy array.

    Parameters:
    - video_path: The path to the video file.

    Returns:
    - A NumPy array containing processed frames of shape (50, 50, 1).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    frames = []  # To hold the processed frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count % 10 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            resized_frame = cv2.resize(gray_frame, (50, 50))  # Resize the frame
            frames.append(resized_frame.reshape(50, 50, 1))  # Add an extra dimension for consistency with the requested shape
        
        frame_count += 1

    cap.release()

    return np.array(frames)


def load_pickle_model(filename):
    """
    Load a model from a pickle file.

    Parameters:
    - filename: The path to the pickle file containing the model.

    Returns:
    - The loaded model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model




def predict_single_image(image_cnn, image_lstm, fashion_model):
    """
    Predicts the class of a single image using the fashion_model.

    Parameters:
    - image_cnn: The image processed for the CNN part of the model.
    - image_lstm: The image processed for the LSTM part of the model.
    - fashion_model: The trained model that will make the prediction.

    Returns:
    - The predicted class of the image.
    """
    # Reshape the images to add a batch dimension, assuming they don't have it
    image_cnn_batch = np.expand_dims(image_cnn, axis=0)
    image_lstm_batch = np.expand_dims(image_lstm, axis=0)

    # Make prediction
    predict_prob = fashion_model.predict([image_cnn_batch, image_lstm_batch])
    
    # Convert probabilities to predicted class
    y_pred = np.argmax(predict_prob, axis=1)

    return y_pred[0]  # Return the predicted class

# Example usage (assuming fashion_model is your trained model):
# predicted_class = predict_single_image(image_cnn, image_lstm, fashion_model)
# print('Predicted class:', predicted_class)


def load_model_with_weights():

    print("loading................")
    # Load the model architecture from JSON
    with open("model_architecture.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    print("architecture loaded ................")

    # Compile the model (if needed)
    # loaded_model.compile(optimizer='adam',
    #                     loss='sparse_categorical_crossentropy',
    #                     metrics=['accuracy'])

    # Load the weights into the model
    loaded_model.load_weights("model_weights.h5")
    print("weights loaded ................")
    return loaded_model





# Example usage
video_path = './videos/fighting.mp4'
frames = extract_and_process_frames(video_path)
print(f"Extracted {frames.shape[0]} frames, each of shape {frames.shape[1:]}.")
frame_0 = frames[0].reshape(-1, 1)
print(frame_0.shape, frames[0].shape)


# Example usage:
# model_filename = './my_model.pkl'
# fashion_model = load_pickle_model(model_filename)

# fashion_model = load_model("./CNN_LSTM.h5")

fashion_model = load_model_with_weights()

if(not fashion_model):
    print("Model not loaded")
    exit()

print(predict_single_image(frames[0], frames[0].reshape(-1, 1), fashion_model))