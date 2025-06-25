"""
Face Detection Module
Contains classes for face detection, blurring, and various processing operations
"""

import cv2
import numpy as np
import os

class FaceDetector:
    """
    Base class for face detection using Haar Cascade Classifier
    """
    def __init__(self, cascade_path=None):
        """
        Initialize the face detector
        
        Args:
            cascade_path (str): Path to custom cascade file. If None, uses OpenCV's built-in cascade
        """
        if cascade_path is None:
            # Use OpenCV's built-in cascade
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:
            self.cascade_path = cascade_path
            
        # Load the cascade classifier
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Validate cascade loading
        if self.face_cascade.empty():
            raise Exception(f"Error: Could not load face cascade classifier from {self.cascade_path}")
        
        print(f"Face detector initialized successfully using: {self.cascade_path}")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            scale_factor (float): How much the image size is reduced at each scale
            min_neighbors (int): How many neighbors each face needs to retain
            min_size (tuple): Minimum face size (width, height)
            
        Returns:
            list: List of face rectangles [(x, y, w, h), ...]
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces

class FaceBlurrer(FaceDetector):
    """
    Class for blurring detected faces
    Inherits from FaceDetector
    """
    
    def __init__(self, cascade_path=None, blur_intensity=51):
        """
        Initialize face blurrer
        
        Args:
            cascade_path (str): Path to cascade file
            blur_intensity (int): Blur kernel size (must be odd number)
        """
        super().__init__(cascade_path)
        
        # Ensure blur intensity is odd
        self.blur_intensity = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        print(f"Blur intensity set to: {self.blur_intensity}")
    
    def blur_faces(self, image, **detection_params):
        """
        Detect and blur faces in an image
        
        Args:
            image: Input image
            **detection_params: Parameters for face detection
            
        Returns:
            tuple: (blurred_image, face_count)
        """
        # Detect faces
        faces = self.detect_faces(image, **detection_params)
        
        # Create copy of original image
        blurred_image = image.copy()
        
        # Blur each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = blurred_image[y:y+h, x:x+w]
            
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(
                face_region, 
                (self.blur_intensity, self.blur_intensity), 
                0
            )
            
            # Replace original face with blurred version
            blurred_image[y:y+h, x:x+w] = blurred_face
        
        return blurred_image, len(faces)
    
    def set_blur_intensity(self, intensity):
        """
        Change blur intensity
        
        Args:
            intensity (int): New blur intensity (will be made odd if even)
        """
        self.blur_intensity = intensity if intensity % 2 == 1 else intensity + 1
        print(f"Blur intensity changed to: {self.blur_intensity}")

class FaceAnnotator(FaceDetector):
    """
    Class for drawing rectangles around detected faces
    Inherits from FaceDetector
    """
    
    def __init__(self, cascade_path=None, rectangle_color=(255, 0, 0), thickness=2):
        """
        Initialize face annotator
        
        Args:
            cascade_path (str): Path to cascade file
            rectangle_color (tuple): BGR color for rectangles
            thickness (int): Rectangle line thickness
        """
        super().__init__(cascade_path)
        self.rectangle_color = rectangle_color
        self.thickness = thickness
    
    def draw_face_rectangles(self, image, **detection_params):
        """
        Detect faces and draw rectangles around them
        
        Args:
            image: Input image
            **detection_params: Parameters for face detection
            
        Returns:
            tuple: (annotated_image, face_count)
        """
        # Detect faces
        faces = self.detect_faces(image, **detection_params)
        
        # Create copy of original image
        annotated_image = image.copy()
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                annotated_image, 
                (x, y), 
                (x + w, y + h), 
                self.rectangle_color, 
                self.thickness
            )
            
            # Add face number label
            cv2.putText(
                annotated_image,
                f"Face {len([f for f in faces if f[0] <= x])}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.rectangle_color,
                2
            )
        
        return annotated_image, len(faces)
    
    def set_rectangle_style(self, color=(255, 0, 0), thickness=2):
        """
        Change rectangle appearance
        
        Args:
            color (tuple): BGR color
            thickness (int): Line thickness
        """
        self.rectangle_color = color
        self.thickness = thickness

class AdvancedFaceProcessor(FaceDetector):
    """
    Advanced face processor with multiple effects
    """
    
    def __init__(self, cascade_path=None):
        super().__init__(cascade_path)
        self.effects = {
            'blur': self._apply_blur,
            'pixelate': self._apply_pixelate,
            'black_box': self._apply_black_box,
            'emoji': self._apply_emoji
        }
    
    def _apply_blur(self, face_region, intensity=51):
        """Apply Gaussian blur to face region"""
        kernel_size = intensity if intensity % 2 == 1 else intensity + 1
        return cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
    
    def _apply_pixelate(self, face_region, pixel_size=20):
        """Apply pixelation effect to face region"""
        h, w = face_region.shape[:2]
        
        # Resize down and then up to create pixelation effect
        temp = cv2.resize(face_region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_black_box(self, face_region, opacity=0.8):
        """Apply black box over face region"""
        overlay = np.zeros_like(face_region)
        return cv2.addWeighted(face_region, 1-opacity, overlay, opacity, 0)
    
    def _apply_emoji(self, face_region, emoji_type='smile'):
        """Apply emoji over face (simplified version)"""
        h, w = face_region.shape[:2]
        center = (w//2, h//2)
        
        # Create a simple smiley face
        emoji_face = face_region.copy()
        
        # Face circle (yellow)
        cv2.circle(emoji_face, center, min(w, h)//3, (0, 255, 255), -1)
        
        # Eyes
        eye_offset = min(w, h)//8
        cv2.circle(emoji_face, (center[0]-eye_offset, center[1]-eye_offset//2), 
                  eye_offset//3, (0, 0, 0), -1)
        cv2.circle(emoji_face, (center[0]+eye_offset, center[1]-eye_offset//2), 
                  eye_offset//3, (0, 0, 0), -1)
        
        # Mouth
        cv2.ellipse(emoji_face, center, (eye_offset, eye_offset//2), 
                   0, 0, 180, (0, 0, 0), 3)
        
        return emoji_face
    
    def apply_effect(self, image, effect='blur', **effect_params):
        """
        Apply specified effect to detected faces
        
        Args:
            image: Input image
            effect (str): Effect type ('blur', 'pixelate', 'black_box', 'emoji')
            **effect_params: Parameters for the effect
            
        Returns:
            tuple: (processed_image, face_count)
        """
        if effect not in self.effects:
            raise ValueError(f"Unknown effect: {effect}. Available: {list(self.effects.keys())}")
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Create copy of original image
        processed_image = image.copy()
        
        # Apply effect to each face
        for (x, y, w, h) in faces:
            face_region = processed_image[y:y+h, x:x+w]
            processed_face = self.effects[effect](face_region, **effect_params)
            processed_image[y:y+h, x:x+w] = processed_face
        
        return processed_image, len(faces)

class FaceDetectionUtils:
    """
    Utility functions for face detection operations
    """
    
    @staticmethod
    def validate_image_path(image_path):
        """
        Validate if image path exists and is readable
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            bool: True if valid, False otherwise
        """
        return os.path.exists(image_path) and os.path.isfile(image_path)
    
    @staticmethod
    def load_image(image_path):
        """
        Load image with error handling
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            numpy.ndarray or None: Loaded image or None if failed
        """
        if not FaceDetectionUtils.validate_image_path(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        return image
    
    @staticmethod
    def save_image(image, output_path):
        """
        Save image with error handling
        
        Args:
            image: Image to save
            output_path (str): Path to save image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = cv2.imwrite(output_path, image)
            if success:
                print(f"Image saved successfully to: {output_path}")
                return True
            else:
                print(f"Error: Failed to save image to {output_path}")
                return False
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def display_images(images, titles=None, wait_key=True):
        """
        Display multiple images in separate windows
        
        Args:
            images (list): List of images to display
            titles (list): List of window titles
            wait_key (bool): Whether to wait for key press
        """
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(images))]
        
        for img, title in zip(images, titles):
            cv2.imshow(title, img)
        
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    @staticmethod
    def get_detection_stats(faces):
        """
        Get statistics about detected faces
        
        Args:
            faces: Array of face rectangles
            
        Returns:
            dict: Statistics dictionary
        """
        if len(faces) == 0:
            return {"count": 0, "avg_size": 0, "total_area": 0}
        
        areas = [w * h for (x, y, w, h) in faces]
        
        return {
            "count": len(faces),
            "avg_size": np.mean(areas),
            "total_area": sum(areas),
            "min_size": min(areas),
            "max_size": max(areas)
        }