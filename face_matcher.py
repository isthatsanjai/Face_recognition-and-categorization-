import os
import glob
from deepface import DeepFace
import cv2 # OpenCV for optionally displaying images or drawing boxes
import numpy as np

# --- Configuration ---
# You can try different models and detector backends
# Common models: "VGG-Face", "Facenet", "Facenet512", "ArcFace", "Dlib", "SFace"
# Common detector_backends: "opencv", "ssd", "dlib", "mtcnn", "retinaface"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "mtcnn" # MTCNN is generally good for accuracy
# The similarity threshold can be tuned. Lower means stricter.
# This threshold is specific to the distance metric used by the model (e.g., cosine for VGG-Face)
# DeepFace.verify will use the model's default threshold, but we can override.
# For VGG-Face, cosine distance < 0.40 is often a good starting point for "same person"
SIMILARITY_THRESHOLD = 0.80 # Adjust based on testing

# --- Helper Functions ---

def get_reference_embedding(image_path, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND):
    """
    Detects a face in the reference image and returns its embedding.
    Returns None if no face is found or multiple faces are detected (for simplicity).
    """
    print(f"\n[INFO] Processing reference image: {image_path}")
    try:
        # DeepFace.represent can handle detection and embedding in one step.
        # It returns a list of dictionaries, one for each detected face.
        # For the reference image, we expect one clear face.
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True # Raise error if no face found
        )

        if not embedding_objs:
            print(f"[WARNING] No face detected in reference image: {image_path}")
            return None

        if len(embedding_objs) > 1:
            print(f"[WARNING] Multiple faces ({len(embedding_objs)}) detected in reference image: {image_path}. Using the first one.")
            # Optionally, you could implement logic to pick the largest face or ask the user.
        
        # 'embedding' is the feature vector
        reference_embedding = embedding_objs[0]["embedding"]
        print(f"[INFO] Reference embedding generated successfully. Vector length: {len(reference_embedding)}")
        return reference_embedding

    except ValueError as e: # Handles enforce_detection=True error if no face found
        print(f"[ERROR] Could not process reference image {image_path}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while processing reference image {image_path}: {e}")
        return None

def get_event_face_embeddings(event_photos_dir, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND):
    """
    Processes all images in the event_photos_dir, detects faces, and extracts their embeddings.
    Returns a list of dictionaries, each containing:
        {'image_path': path_to_image, 'embeddings': [emb1, emb2, ...], 'facial_areas': [area1, area2,...]}
    """
    print(f"\n[INFO] Processing event photos in directory: {event_photos_dir}")
    event_data = []
    # Look for common image types
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(event_photos_dir, ext)))

    if not image_paths:
        print(f"[WARNING] No images found in directory: {event_photos_dir}")
        return event_data

    print(f"[INFO] Found {len(image_paths)} images to process.")

    for image_path in image_paths:
        print(f"  Processing: {os.path.basename(image_path)}...")
        try:
            # enforce_detection=False: if no face, it returns an empty list for this image
            # we want to process images even if no faces are found, so we can report it.
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )

            current_image_embeddings = []
            current_image_facial_areas = []
            if embedding_objs: # If any faces were detected
                for obj in embedding_objs:
                    current_image_embeddings.append(obj["embedding"])
                    current_image_facial_areas.append(obj["facial_area"]) # Bounding box: {'x': 268, 'y': 90, 'w': 172, 'h': 209}
            if not current_image_embeddings:
                print(f"    No faces detected in {os.path.basename(image_path)}")
            else:
                print(f"    Detected {len(current_image_embeddings)} face(s) in {os.path.basename(image_path)}")

            event_data.append({
                "image_path": image_path,
                "embeddings": current_image_embeddings,
                "facial_areas": current_image_facial_areas
            })

        except Exception as e:
            print(f"[ERROR] Could not process event image {image_path}: {e}")
            # Optionally, add this image with an error status to event_data
            event_data.append({
                "image_path": image_path,
                "embeddings": [],
                "facial_areas": [],
                "error": str(e)
            })
    return event_data

def find_matches(reference_embedding, event_data, threshold=SIMILARITY_THRESHOLD, model_name=MODEL_NAME):
    """
    Compares the reference embedding against all face embeddings in the event_data.
    Returns a list of image paths that contain a match.
    """
    print(f"\n[INFO] Finding matches using threshold: {threshold} (lower is stricter)")
    matched_image_paths = set() # Use a set to store unique image paths

    if reference_embedding is None:
        print("[ERROR] Reference embedding is None. Cannot perform matching.")
        return list(matched_image_paths)

    for data_item in event_data:
        if "error" in data_item: # Skip images that had processing errors
            continue

        for event_embedding, facial_area in zip(data_item["embeddings"], data_item["facial_areas"]):
            try:
                # DeepFace.verify performs the comparison.
                # It can take two image paths, or pre-calculated embeddings.
                # result is a dictionary: e.g. {'verified': True, 'distance': 0.25, 'threshold': 0.40, ...}
                result = DeepFace.verify(
                    img1_path=reference_embedding, # Can pass embedding directly
                    img2_path=event_embedding,     # Can pass embedding directly
                    model_name=model_name,
                    distance_metric="cosine" # VGG-Face typically uses cosine
                    # detector_backend is not needed here as we are passing embeddings
                )
                
                # We use our own threshold for more control, but verify's threshold is a good guide
                # Note: result['verified'] will be based on the model's default threshold.
                # We check distance against our SIMILARITY_THRESHOLD.
                if result["distance"] <= threshold:
                    print(f"  MATCH FOUND: {os.path.basename(data_item['image_path'])} (Distance: {result['distance']:.4f})")
                    matched_image_paths.add(data_item["image_path"])
                    
                    # Optional: Display the matched image with bounding box
                    # img = cv2.imread(data_item['image_path'])
                    # x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.imshow(f"Match: {os.path.basename(data_item['image_path'])}", img)
                    # cv2.waitKey(1000) # Display for 1 second
                    # cv2.destroyWindow(f"Match: {os.path.basename(data_item['image_path'])}")

            except Exception as e:
                print(f"[ERROR] Error during verification for {data_item['image_path']}: {e}")
                # This can happen if embeddings are not in the expected format, though less likely
                # when both are generated by DeepFace.represent.

    return list(matched_image_paths)

# --- Main Execution ---
if __name__ == "__main__":
    # Define paths (adjust as needed)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "reference_image", "me.jpeg") # CHANGE 'me.jpg' to your reference image filename
    EVENT_PHOTOS_DIR = os.path.join(BASE_DIR, "event_photos")

    print("--- Face Recognition for Event Photos ---")

    # 1. Get embedding for the reference image
    ref_embedding = get_reference_embedding(REFERENCE_IMAGE_PATH)

    if ref_embedding:
        # 2. Get embeddings for all faces in event photos
        all_event_data = get_event_face_embeddings(EVENT_PHOTOS_DIR)

        # 3. Find and print matches
        matched_photos = find_matches(ref_embedding, all_event_data)

        if matched_photos:
            print(f"\n[RESULT] The reference person was found in the following {len(matched_photos)} photos:")
            for photo_path in matched_photos:
                print(f"  - {photo_path}")
        else:
            print("\n[RESULT] The reference person was not found in any of the event photos (or an error occurred).")
    else:
        print("\n[RESULT] Could not proceed without a valid reference embedding.")
    
    # cv2.destroyAllWindows() # Close any OpenCV windows if you uncommented display code
    print("\n--- Processing Complete ---")