# Face Recognition Project

This face recognition project uses OpenCV to detect and recognize faces from a webcam stream. It's built using Python and utilizes Haar Cascades for face detection and LBPH (Local Binary Patterns Histograms) for face recognition.

## Prerequisites

Before you run this project, ensure you have Python installed on your system. The project was developed using Python 3.8, but it should be compatible with other Python 3 versions.

## Installation

Clone the repository to your local machine:

git clone https://github.com/yourusername/face-recognition-opencv.git cd face-recognition-opencv

Install the necessary Python packages:

pip install -r requirements.txt


## Usage

To run the project, you need to perform the following steps:

1. **Prepare the Data:**
   - Place your training images in the `data/images/` directory. Ensure that images are organized into subfolders named after the person they represent.

2. **Train the Model:**
   - Navigate to the project directory and run the training script:
     ```
     python src/face_training.py
     ```

3. **Run Face Detection and Recognition:**
   - To start face detection, execute:
     ```
     python src/face_detection.py
     ```
   - To start face recognition, execute:
     ```
     python src/face_recognition.py
     ```
