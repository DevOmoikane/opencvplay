import cv2
import numpy as np
import os
from pathlib import Path
import httpx
import mediapipe as mp
from dotenv import dotenv_values
import base64
import click
import chromadb
from chromadb.config import Settings
import uuid
from rich import print
import traceback
import re
from typing import Dict, Any, Union


config = {
    **dotenv_values(".env"),
    **os.environ,
}


def file_to_base64(file_path):
    """
    Loads a file from the given path and returns its content as a Base64 encoded string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The Base64 encoded string of the file's content, or None if an error occurs.
    """
    try:
        with open(file_path, "rb") as file:  # Open the file in binary read mode
            encoded_bytes = base64.b64encode(file.read())  # Encode the file content
            encoded_string = encoded_bytes.decode("ascii")  # Decode bytes to a UTF-8 string
            return "data:image/jpeg;base64," + encoded_string
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def flatten_dict(nested_dict, parent_key='', sep='.'):
    """
    Flatten a nested dictionary using dot notation
    
    Args:
        nested_dict: The dictionary to flatten
        parent_key: Used for recursion, leave empty
        sep: Separator for nested keys (default: '.')
    
    Returns:
        Flattened dictionary with single level
    """
    items = []
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Recursively flatten dictionaries
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            # Handle lists - convert to indexed keys
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            # Base case: add the key-value pair
            items.append((new_key, value))
    
    return dict(items)


def english_to_chromadb_filter(english_condition: str) -> Dict[str, Any]:
    """
    Translate plain English conditional filtering to ChromaDB filter format.

    Args:
        english_condition: Plain English condition string

    Returns:
        ChromaDB filter dictionary

    Examples:
        >>> english_to_chromadb_filter("page > 10")
        {"page": {"$gt": 10}}

        >>> english_to_chromadb_filter("page >= 5 and page <= 10")
        {"$and": [{"page": {"$gte": 5}}, {"page": {"$lte": 10}}]}

        >>> english_to_chromadb_filter("category == 'science' or year > 2020")
        {"$or": [{"category": {"$eq": "science"}}, {"year": {"$gt": 2020}}]}
    """
    # Normalize the input
    condition = english_condition.strip().lower()

    # Handle compound conditions with AND/OR
    if " and " in condition:
        parts = condition.split(" and ")
        return {
            "$and": [english_to_chromadb_filter(part.strip()) for part in parts]
        }

    if " or " in condition:
        parts = condition.split(" or ")
        return {
            "$or": [english_to_chromadb_filter(part.strip()) for part in parts]
        }

    # Handle single conditions
    return _parse_single_condition(condition)


def _parse_single_condition(condition: str) -> Dict[str, Any]:
    """Parse a single conditional expression."""

    # Regex pattern to match: field operator value
    pattern = r'^(\w+)\s*([<>!=]=?|>=|<=)\s*(.+)$'
    match = re.match(pattern, condition.strip())

    if not match:
        raise ValueError(f"Invalid condition format: {condition}")

    field, operator, value_str = match.groups()

    # Parse the value (handle strings, numbers, and boolean)
    value = _parse_value(value_str.strip())

    # Map operators to ChromaDB operators
    operator_map = {
        '>': '$gt',
        '>=': '$gte',
        '<': '$lt',
        '<=': '$lte',
        '==': '$eq',
        '=': '$eq',
        '!=': '$ne'
    }

    chroma_operator = operator_map.get(operator)
    if not chroma_operator:
        raise ValueError(f"Unsupported operator: {operator}")

    return {field: {chroma_operator: value}}


def _parse_value(value_str: str) -> Union[str, int, float, bool]:
    """Parse and convert value string to appropriate type."""

    # Remove quotes if present
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
            (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]

    # Handle boolean values
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'

    # Handle numbers
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # Return as string if not a number
        return value_str


# Enhanced version with more features
def english_to_chromadb_filter_advanced(english_condition: str) -> Dict[str, Any]:
    """
    Enhanced version with support for parentheses and complex conditions.
    """
    condition = english_condition.strip()

    # Handle parentheses for complex conditions
    if '(' in condition and ')' in condition:
        return _parse_complex_condition(condition)

    # Handle IN conditions (e.g., "category in ['science', 'math']")
    if ' in ' in condition and '[' in condition and ']' in condition:
        return _parse_in_condition(condition)

    # Handle NOT conditions
    if ' not ' in condition:
        return _parse_not_condition(condition)

    return english_to_chromadb_filter(condition)


def _parse_complex_condition(condition: str) -> Dict[str, Any]:
    """Parse conditions with parentheses."""
    # This is a simplified version - you might want to use a proper parser for complex cases
    condition = condition.replace('(', '').replace(')', '')
    return english_to_chromadb_filter(condition)


def _parse_in_condition(condition: str) -> Dict[str, Any]:
    """Parse IN conditions like: field in [value1, value2]"""
    pattern = r'^(\w+)\s+in\s+\[(.+)\]$'
    match = re.match(pattern, condition.strip())

    if match:
        field, values_str = match.groups()
        # Parse the values list
        values = [_parse_value(val.strip()) for val in values_str.split(',')]
        return {field: {"$in": values}}

    raise ValueError(f"Invalid IN condition format: {condition}")


def _parse_not_condition(condition: str) -> Dict[str, Any]:
    """Parse NOT conditions."""
    if ' not in ' in condition:
        pattern = r'^(\w+)\s+not in\s+\[(.+)\]$'
        match = re.match(pattern, condition.strip())
        if match:
            field, values_str = match.groups()
            values = [_parse_value(val.strip()) for val in values_str.split(',')]
            return {field: {"$nin": values}}

    # Handle simple NOT conditions
    condition = condition.replace(' not ', ' != ')
    return english_to_chromadb_filter(condition)


class DeepFace:
    @staticmethod
    def analyze(img_path, actions, enforce_detection):
        """Analyze facial and body features"""
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        base64_img = file_to_base64(img_path)
        
        if actions is None or len(actions) == 0:
            actions = ['age', 'gender', 'race', 'emotion']

        response = httpx.post(
            config['DEEPFACE_URL'] + "/analyze",
            json={
                "img": base64_img,
                "actions": actions,
                "enforce_detection": enforce_detection,
            },
            timeout=None
        )
        
        if response.status_code == 200:
            rj = response.json()
            return rj
        
        raise Exception(f"Response wat not succesfull {response.status_code}: {response.text} ")
        
    def verify(img1_path, img2_path):
        """Verify similarity between two images"""
        base64_img1 = file_to_base64(img1_path)
        base64_img2 = file_to_base64(img2_path)
        
        response = httpx.post(
            config['DEEPFACE_URL'] + "/verify",
            json={
                "img1": base64_img1,
                "img2": base64_img2
            },
            timeout=None
        )
        
        if response.status_code == 200:
            return response.json()
        
        return None
        
    def represent(img_path, model_name, enforce_detection=True):
        """Extract facial features"""
        if not os.path.exists(img_path):
            return None
        
        base64_img = file_to_base64(img_path)
        
        response = httpx.post(
            config['DEEPFACE_URL'] + "/represent",
            json={
                "img": base64_img,
                "model_name": model_name,
                "enforce_detection": enforce_detection,
            },
            timeout=None
        )

        if response.status_code == 200:
            return response.json()
        
        return None

class AdvancedPersonFinder:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.reference_features = {}
        self.client = None
        self.person_collection = None

    def connect_chroma(self):
        self.client = chromadb.HttpClient(host=config['CHROMA_HOST'], port=config['CHROMA_PORT'], ssl=False, headers=None, settings=Settings(anonymized_telemetry=False))
        self.person_collection = self.client.get_or_create_collection(name='persons')

    def is_person_in_db(self, file_name, embedding):
        if self.client is None:
            return False
        results = self.person_collection.query(query_embeddings=[embedding], where={ "file_name": file_name } ,n_results=1, include=["metadatas", "distances"])
        if len(results['ids'][0]) > 0:
            return True
        return False

    def search_db_embedding(self, embedding, threshold=0.5):
        if self.client is None:
            return None
        results = self.person_collection.query(query_embeddings=[embedding], n_results=1, include=["metadatas", "distances"])

    def add_to_db(self, embedding, metadata):
        if self.client is None:
            return False
        curr_id = str(uuid.uuid4())
        self.person_collection.add(embeddings=[embedding], metadatas=[metadata], ids=[curr_id])
        return True

    
    def extract_reference_features(self, reference_image_path):
        """Extract facial and body features from reference image"""
        # Facial analysis with DeepFace
        response = DeepFace.analyze(
            img_path=reference_image_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )
        
        if 'results' not in response or len(response['results']) == 0:
            return None
        
        facial_analysis = response['results'][0]
        # Body pose estimation
        image = cv2.imread(reference_image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_image)
        
        self.reference_features = {
            'facial': facial_analysis,
            'pose_landmarks': pose_results.pose_landmarks,
            'embedding': self.get_facial_embedding(reference_image_path)
        }
        
        return self.reference_features
    
    def get_facial_embedding(self, image_path):
        """Get facial embedding vector"""
        try:
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet',
                enforce_detection=False
            )
            return embedding_objs['results'][0]['embedding']
        except:
            return None
    
    def calculate_similarity_score(self, candidate_features):
        """Calculate overall similarity score"""
        scores = []
        
        # Facial similarity
        if self.reference_features.get('embedding') and candidate_features.get('embedding'):
            ref_embedding = np.array(self.reference_features['embedding'])
            cand_embedding = np.array(candidate_features['embedding'])
            facial_similarity = 1 - (np.linalg.norm(ref_embedding - cand_embedding) / 100)
            scores.append(facial_similarity)
        
        # Demographic similarity
        if self.reference_features.get('facial') and candidate_features.get('facial'):
            demo_score = self.calculate_demographic_similarity(
                self.reference_features['facial'],
                candidate_features['facial']
            )
            scores.append(demo_score)
        
        # Body proportions similarity
        body_score = self.calculate_body_similarity(candidate_features.get('pose_landmarks'))
        scores.append(body_score)
        
        return np.mean(scores)
    
    def calculate_demographic_similarity(self, ref_demo, cand_demo):
        """Calculate demographic similarity score"""
        score = 0
        weight = 0
        
        # Age similarity (within 10 years)
        age_diff = abs(ref_demo['age'] - cand_demo['age'])
        age_score = max(0, 1 - (age_diff / 30))
        score += age_score * 0.3
        weight += 0.3
        
        # Gender similarity
        if ref_demo['dominant_gender'] == cand_demo['dominant_gender']:
            score += 1 * 0.3
        weight += 0.3
        
        # Race similarity
        if ref_demo['dominant_race'] == cand_demo['dominant_race']:
            score += 1 * 0.4
        weight += 0.4
        
        return score / weight if weight > 0 else 0
    
    def calculate_body_similarity(self, candidate_landmarks):
        """Calculate body proportions similarity"""
        if not self.reference_features.get('pose_landmarks') or not candidate_landmarks:
            return 0.5  # Neutral score if no body data
        
        # Compare key body proportions
        ref_landmarks = self.reference_features['pose_landmarks']
        
        # Example: Compare height-to-width ratio
        ref_height = self.get_vertical_distance(ref_landmarks)
        ref_width = self.get_horizontal_distance(ref_landmarks)
        cand_height = self.get_vertical_distance(candidate_landmarks)
        cand_width = self.get_horizontal_distance(candidate_landmarks)
        
        if ref_height > 0 and ref_width > 0 and cand_height > 0 and cand_width > 0:
            ref_ratio = ref_height / ref_width
            cand_ratio = cand_height / cand_width
            ratio_similarity = 1 - abs(ref_ratio - cand_ratio) / max(ref_ratio, cand_ratio)
            return max(0, ratio_similarity)
        
        return 0.5
    
    def get_vertical_distance(self, landmarks):
        """Calculate vertical distance between head and feet"""
        if landmarks.landmark[self.mp_pose.PoseLandmark.NOSE] and \
           landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            return abs(nose.y - ankle.y)
        return 0
    
    def get_horizontal_distance(self, landmarks):
        """Calculate horizontal distance between shoulders"""
        if landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER] and \
           landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            return abs(left_shoulder.x - right_shoulder.x)
        return 0
    
    def find_similar_persons(self, photos_directory, min_similarity=0.5):
        similar_photos = []
        
        for file_path in Path(photos_directory).glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    print(f"Processing: {file_path}")
                    
                    # Extract candidate features
                    candidate_features = {}
                    
                    # Facial features
                    response = DeepFace.analyze(
                        img_path=str(file_path),
                        actions=['age', 'gender', 'race', 'emotion'],
                        enforce_detection=False
                    )
                    facial_analysis = response['results'][0]
                    candidate_features['facial'] = facial_analysis
                    
                    # Facial embedding
                    candidate_features['embedding'] = self.get_facial_embedding(str(file_path))
                    
                    # Body features
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pose_results = self.pose.process(rgb_image)
                        candidate_features['pose_landmarks'] = pose_results.pose_landmarks
                    
                    #TODO: saved data into chromadb for faster search

                    # Calculate similarity
                    similarity = self.calculate_similarity_score(candidate_features)
                    
                    if similarity >= min_similarity:
                        similar_photos.append({
                            'file_path': str(file_path),
                            'similarity': similarity,
                            'features': candidate_features
                        })
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e} in {traceback.format_stack}")
        
        # Sort by similarity
        similar_photos.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_photos
    
    def add_person_to_db(self, image_path):

        try:
            file_name = os.path.basename(image_path)
            print(f"Processing: {str(file_name)}")

            # Facial embedding
            embedding = self.get_facial_embedding(str(image_path))
            
            if self.is_person_in_db(file_name, embedding):
                raise Exception("Person already in DB")
                # pass

            # Facial features
            response = DeepFace.analyze(
                img_path=str(image_path),
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            facial_analysis = response['results'][0]
            
            # Body features
            image = cv2.imread(str(image_path))
            pose_results = None
            if image is not None:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb_image)
                landmarks = pose_results.pose_landmarks

            landmarks_array = {}
            if hasattr(landmarks, 'landmark') and landmarks.landmark is not None:
                for poselandmark in self.mp_pose.PoseLandmark:
                    try:
                        landmark = landmarks.landmark[poselandmark]
                        landmarks_array[poselandmark.name] = {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility
                        }
                    except:
                        pass

            metadatas = {"file_name": file_name, **facial_analysis, "landmarks": landmarks_array}
            flat_metadatas = flatten_dict(metadatas, sep='_')
            cleaned_metadatas = {k: v if v is not None else "" for k, v in flat_metadatas.items()}
            print(f"Metadatas = {cleaned_metadatas} ")
            #add the data to the db
            self.add_to_db(embedding, cleaned_metadatas)
        except Exception as e:
            print(f"Error processing {image_path}: {e} in {traceback.format_exception(e)}")

    def add_path_to_db(self, images_path):
        for file_path in Path(images_path).glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.add_person_to_db(file_path)

    def search
    def search_person_by_metadata(self, metadata_filter):
        if self.client is None:
            return None
        results = self.person_collection.query(
            query_texts=[metadata_filter],
            where=metadata_filter,
            n_results=1,
            include=["metadatas", "distances"]
        )
        if len(results['ids'][0]) > 0:


@click.command()
@click.option('--base-image', required=False)
@click.option('--image-path', required=False)
@click.option('--min-similarity', required=False, default=0.6)
@click.option('--show-results-imgs/--no-show-results-imgs', default=False)
@click.option('--search-sentence', required=False)
@click.option('--fill-db/--no-fill-db', default=False)
def main(base_image, image_path, min_similarity, show_results_imgs, search_sentence, fill_db):
    finder = AdvancedPersonFinder()
    finder.connect_chroma()
    if base_image is not None:
        finder.extract_reference_features(base_image)
        similar_persons = finder.find_similar_persons(image_path, min_similarity)
        if show_results_imgs:
            for i, person in enumerate(similar_persons, 1):
                print(f"{i}. {person['file_path']} - Similarity: {person['similarity']:.2%}")
                img = cv2.imread(person['file_path'])
                if img is not None:
                    cv2.imshow(f"Person", img)
                    cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            for i, person in enumerate(similar_persons, 1):
                print(f"{i}. {person['file_path']} - Similarity: {person['similarity']:.2%}")
    elif fill_db and image_path is not None:
        finder.add_path_to_db(image_path)
        # finder.add_person_to_db(os.path.join(image_path, "IMG_20250925_123933_316.jpg"))
    elif search_sentence is not None:
        pass


if __name__ == "__main__":
    main()
