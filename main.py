"""
Main Application - Face Detection and Processing
Imports and uses classes from face_detection.py module
"""

import cv2
import os
from face_detection import (
    FaceDetector, 
    FaceBlurrer, 
    FaceAnnotator, 
    AdvancedFaceProcessor, 
    FaceDetectionUtils
)

class FaceProcessingApp:
    """
    Main application class that coordinates different face processing operations
    """
    
    def __init__(self):
        """Initialize the application with different processors"""
        print("Initializing Face Processing Application...")
        
        try:
            # Initialize different processors
            self.face_detector = FaceDetector()
            self.face_blurrer = FaceBlurrer(blur_intensity=51)
            self.face_annotator = FaceAnnotator(rectangle_color=(0, 255, 0), thickness=3)
            self.advanced_processor = AdvancedFaceProcessor()
            
            print("✓ All processors initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing processors: {e}")
            raise
    
    def process_single_image(self, image_path, operation='blur', output_path=None, **params):
        """
        Process a single image with specified operation
        
        Args:
            image_path (str): Path to input image
            operation (str): Type of operation ('blur', 'annotate', 'detect', 'advanced')
            output_path (str): Path to save result (optional)
            **params: Additional parameters for processing
        """
        print(f"\n🖼️  Processing image: {image_path}")
        print(f"📋 Operation: {operation}")
        
        # Load image
        image = FaceDetectionUtils.load_image(image_path)
        if image is None:
            return
        
        # Process based on operation type
        try:
            if operation == 'blur':
                result_image, face_count = self.face_blurrer.blur_faces(image, **params)
                operation_text = f"Blurred {face_count} face(s)"
                
            elif operation == 'annotate':
                result_image, face_count = self.face_annotator.draw_face_rectangles(image, **params)
                operation_text = f"Annotated {face_count} face(s)"
                
            elif operation == 'detect':
                faces = self.face_detector.detect_faces(image, **params)
                stats = FaceDetectionUtils.get_detection_stats(faces)
                print(f"📊 Detection Stats: {stats}")
                result_image = image.copy()
                face_count = len(faces)
                operation_text = f"Detected {face_count} face(s)"
                
            elif operation == 'advanced':
                effect = params.get('effect', 'blur')
                result_image, face_count = self.advanced_processor.apply_effect(image, effect, **params)
                operation_text = f"Applied {effect} effect to {face_count} face(s)"
                
            else:
                print(f"❌ Unknown operation: {operation}")
                return
            
            print(f"✓ {operation_text}")
            
            # Save result if output path provided
            if output_path:
                FaceDetectionUtils.save_image(result_image, output_path)
            
            # Display results
            FaceDetectionUtils.display_images(
                [image, result_image], 
                ['Original', f'Result ({operation})']
            )
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
    
    def process_webcam(self, operation='blur', **params):
        """
        Process webcam feed in real-time
        
        Args:
            operation (str): Type of operation
            **params: Additional parameters
        """
        print(f"\n📹 Starting webcam processing with operation: {operation}")
        print("Controls: 'q' = quit, 'b' = blur, 'a' = annotate, 'p' = pixelate, 'e' = emoji")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        current_operation = operation
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            try:
                # Process frame based on current operation
                if current_operation == 'blur':
                    processed_frame, face_count = self.face_blurrer.blur_faces(frame)
                elif current_operation == 'annotate':
                    processed_frame, face_count = self.face_annotator.draw_face_rectangles(frame)
                elif current_operation == 'pixelate':
                    processed_frame, face_count = self.advanced_processor.apply_effect(
                        frame, 'pixelate', pixel_size=15
                    )
                elif current_operation == 'emoji':
                    processed_frame, face_count = self.advanced_processor.apply_effect(
                        frame, 'emoji'
                    )
                else:
                    processed_frame = frame
                    face_count = 0
                
                # Add status text
                cv2.putText(
                    processed_frame, 
                    f"Mode: {current_operation.title()} | Faces: {face_count}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow('Face Processing - Webcam', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    current_operation = 'blur'
                    print("📝 Switched to blur mode")
                elif key == ord('a'):
                    current_operation = 'annotate'
                    print("📝 Switched to annotate mode")
                elif key == ord('p'):
                    current_operation = 'pixelate'
                    print("📝 Switched to pixelate mode")
                elif key == ord('e'):
                    current_operation = 'emoji'
                    print("📝 Switched to emoji mode")
                
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Webcam session ended")
    
    def process_video_file(self, video_path, output_path, operation='blur', **params):
        """
        Process a video file
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to output video
            operation (str): Type of operation
            **params: Additional parameters
        """
        print(f"\n🎬 Processing video: {video_path}")
        print(f"💾 Output will be saved to: {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                if operation == 'blur':
                    processed_frame, _ = self.face_blurrer.blur_faces(frame)
                elif operation == 'annotate':
                    processed_frame, _ = self.face_annotator.draw_face_rectangles(frame)
                elif operation.startswith('advanced_'):
                    effect = operation.split('_')[1]
                    processed_frame, _ = self.advanced_processor.apply_effect(frame, effect)
                else:
                    processed_frame = frame
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                
                # Show progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
            
            print(f"✅ Video processing complete! Processed {frame_count} frames")
            
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
        
        finally:
            cap.release()
            out.release()
    
    def batch_process_images(self, image_folder, output_folder, operation='blur', **params):
        """
        Process multiple images in a folder
        
        Args:
            image_folder (str): Folder containing input images
            output_folder (str): Folder to save processed images
            operation (str): Type of operation
            **params: Additional parameters
        """
        print(f"\n📁 Batch processing images from: {image_folder}")
        print(f"💾 Results will be saved to: {output_folder}")
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Supported image extensions
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(supported_extensions)]
        
        if not image_files:
            print("❌ No supported image files found in the folder")
            return
        
        print(f"📊 Found {len(image_files)} image(s) to process")
        
        successful = 0
        failed = 0
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(image_folder, filename)
            output_filename = f"{operation}_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"⏳ Processing {i}/{len(image_files)}: {filename}")
            
            try:
                image = FaceDetectionUtils.load_image(input_path)
                if image is None:
                    failed += 1
                    continue
                
                # Process based on operation
                if operation == 'blur':
                    result_image, face_count = self.face_blurrer.blur_faces(image, **params)
                elif operation == 'annotate':
                    result_image, face_count = self.face_annotator.draw_face_rectangles(image, **params)
                else:
                    result_image, face_count = self.advanced_processor.apply_effect(image, operation, **params)
                
                # Save result
                if FaceDetectionUtils.save_image(result_image, output_path):
                    print(f"  ✓ {face_count} face(s) processed")
                    successful += 1
                else:
                    failed += 1
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                failed += 1
        
        print(f"\n📊 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")

def main():
    """Main function to run the application"""
    
    print("=" * 60)
    print("🎭 FACE DETECTION AND PROCESSING APPLICATION")
    print("=" * 60)
    
    try:
        # Initialize the application
        app = FaceProcessingApp()
        
        # Configuration - Change these paths as needed
        IMAGE_PATH = r'C:\Users\HC\OneDrive - Higher Education Commission\Desktop\COMPUTER  VISION\FB_IMG_1666624016389.jpg'
        OUTPUT_IMAGE = 'processed_image.jpg'
        VIDEO_PATH = 'input_video.mp4'
        OUTPUT_VIDEO = 'processed_video.mp4'
        
        # Example usage - Uncomment the operations you want to use:
        
        # 1. Process single image with blur
        if os.path.exists(IMAGE_PATH):
            print("\n🔸 Processing single image with blur...")
            app.process_single_image(IMAGE_PATH, 'blur', OUTPUT_IMAGE)
        
        # 2. Process single image with annotation
        if os.path.exists(IMAGE_PATH):
             print("\n🔸 Processing single image with annotation...")
             app.process_single_image(IMAGE_PATH, 'annotate', 'annotated_image.jpg')
        
        # 3. Advanced processing with pixelation
        if os.path.exists(IMAGE_PATH):
             print("\n🔸 Processing single image with pixelation...")
             app.process_single_image(IMAGE_PATH, 'advanced', 'pixelated_image.jpg', effect='pixelate', pixel_size=20)
        
        # 4. Process webcam feed
        print("\n🔸 Starting webcam processing...")
        app.process_webcam('blur')
        
        # 5. Process video file
        # if os.path.exists(VIDEO_PATH):
        #     print("\n🔸 Processing video file...")
        #     app.process_video_file(VIDEO_PATH, OUTPUT_VIDEO, 'blur')
        
        # 6. Batch process images
        # print("\n🔸 Batch processing images...")
        # app.batch_process_images('input_images/', 'output_images/', 'blur')
        
        print("\n✅ Application completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()