# Event Face Matcher 📸

## Project Overview

Event Face Matcher is a Python-based proof-of-concept application designed to automatically identify individuals in event photos. Attendees can provide a reference selfie, and the system will then attempt to find all photos from an event in which they appear.

This project serves as a practical exploration into face detection and recognition technologies, utilizing the `DeepFace` library and common computer vision techniques.

## Features (Current & Planned)

*   **Face Detection:** Identifies faces in uploaded event photographs using various detector backends (e.g., MTCNN, SSD).
*   **Face Embedding:** Generates numerical vector representations (embeddings) for each detected face using pre-trained deep learning models (e.g., VGG-Face, FaceNet).
*   **Reference Matching:** Compares the embedding of an attendee's reference photo against all face embeddings from event photos to find matches.
*   **Command-Line Interface:** Currently operates as a Python script for local processing.

**(Planned Features - for a full application):**
*   Web interface for studios to upload event albums.
*   Web interface for attendees to upload reference photos and contact details.
*   Automated notification (email/WhatsApp) to attendees with their photos.
*   Asynchronous processing for handling large photo sets.
*   Database storage for embeddings and event data.

## Core Technologies Used

*   **Python 3.9+**
*   **DeepFace:** A lightweight Python framework for face recognition and facial attribute analysis, wrapping state-of-the-art models like VGG-Face, FaceNet, ArcFace, etc.
*   **OpenCV (`opencv-python`):** For image manipulation and as a backend for some face detectors.
*   **NumPy:** For numerical operations, especially with embeddings.
*   **tqdm:** For progress bars during long processing tasks.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file. See below.)*

## How to Run (Current Script)

1.  **Prepare your images:**
    *   Create a directory named `reference_images/` in the project root.
    *   Place **one clear selfie** of the person you want to find into this directory (e.g., `me.jpg`).
    *   Create a directory named `event_photos/` in the project root.
    *   Place the event photos you want to process into this directory.

2.  **Update script configuration (if needed):**
    *   Open `face_matcher.py`.
    *   Ensure `REFERENCE_IMAGE_PATH` points to your reference image (e.g., `os.path.join(BASE_DIR, "reference_images", "me.jpg")`).
    *   You can experiment with `MODEL_NAME`, `DETECTOR_BACKEND`, and `SIMILARITY_THRESHOLD` for different results.

3.  **Run the script:**
    ```bash
    python face_matcher.py
    ```

4.  **Check results:**
    *   The script will print the paths of event photos where the reference person is believed to be found.
    *   If you've enabled the detection visualization code, images with green boxes around detected faces will be saved in `event_photos/detections_output/`.

## Creating `requirements.txt`

To make it easy for others (and your future self) to install dependencies, create a `requirements.txt` file:

1.  Make sure your virtual environment is activated.
2.  Run:
    ```bash
    pip freeze > requirements.txt
    ```
3.  Add and commit this `requirements.txt` file to your repository.

It will contain lines like:
Use code with caution.
Markdown
deepface==0.0.79
numpy==1.26.0
opencv-python==4.8.0.76
tqdm==4.66.1
tensorflow==... # or torch, depending on DeepFace backend
... and other dependencies
## Project Structure
.
├── .gitignore
├── face_matcher.py # Main script for face matching
├── reference_images/ # (Ignored by Git) Place your reference selfies here
│ └── me.jpg
├── event_photos/ # (Ignored by Git) Place event photos here
│ ├── detections_output/ # (Created by script) Output for visualized detections
│ └── event_photo1.jpg
├── requirements.txt # Python dependencies
└── README.md # This file
## To-Do / Future Enhancements

*   [ ] Develop a web backend (e.g., using Flask or FastAPI).
*   [ ] Create user interfaces for studios and attendees.
*   [ ] Implement database storage for efficient querying of embeddings (e.g., PostgreSQL with pgvector).
*   [ ] Integrate asynchronous task queues (e.g., Celery with Redis) for non-blocking processing.
*   [ ] Add a notification system (email/WhatsApp).
*   [ ] Improve accuracy and robustness (e.g., better handling of difficult poses, lighting).
*   [ ] Add more comprehensive error handling and logging.
*   [ ] Write unit and integration tests.

## Contributing

This is currently a solo project for learning and portfolio purposes. However, suggestions and feedback are welcome! Please open an issue to discuss potential changes or features.

## License

*(Choose a license if you wish, e.g., MIT. For now, you can leave this out or state "This project is unlicensed." or "All rights reserved.")*

---
