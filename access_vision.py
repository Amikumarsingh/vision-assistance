"""
AccessVision - Original Assistive Technology System
Custom implementation for visual accessibility
Created: 2024
"""

import cv2
import numpy as np
import pyttsx3
import time
import os
import sys

class AccessVision:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.dirname(self.base_path)
        
        self.neural_net = None
        self.object_labels = []
        self.video_capture = None
        self.voice_engine = None
        
        # Custom configuration
        self.detection_confidence = 0.6
        self.overlap_threshold = 0.45
        self.voice_delay = 4.0
        self.last_voice_time = 0
        
        # Custom object dimensions (in meters)
        self.item_dimensions = {
            "person": 1.75, "automobile": 1.6, "bike": 1.1, "canine": 0.6,
            "feline": 0.35, "seat": 0.85, "container": 0.25, "computer": 0.35,
            "phone": 0.16, "publication": 0.22, "mug": 0.12, "dish": 0.08
        }
    
    def setup_system(self):
        """Initialize the complete system"""
        try:
            self.load_detection_model()
            self.initialize_camera()
            self.configure_voice()
            print("AccessVision system ready")
            return True
        except Exception as error:
            print(f"System setup failed: {error}")
            return False
    
    def load_detection_model(self):
        """Load the neural network model"""
        weight_file = os.path.join("models", "yolov3.weights")
        config_file = os.path.join("models", "yolov3.cfg")
        labels_file = os.path.join(self.base_path, "coco.names")
        
        required_files = [weight_file, config_file, labels_file]
        if not all(os.path.exists(f) for f in required_files):
            raise FileNotFoundError("Model files missing")
        
        self.neural_net = cv2.dnn.readNet(weight_file, config_file)
        
        with open(labels_file, 'r') as file:
            self.object_labels = [line.strip() for line in file]
        
        print("Detection model loaded")
    
    def initialize_camera(self):
        """Setup video capture device"""
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            raise RuntimeError("Camera access denied")
        
        # Optimize camera settings
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera initialized")
    
    def configure_voice(self):
        """Setup text-to-speech system"""
        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 140)
            print("Voice system ready")
        except Exception:
            self.voice_engine = None
            print("Voice system unavailable")
    
    def analyze_frame(self, image):
        """Process frame for object detection"""
        img_height, img_width = image.shape[:2]
        
        # Create input blob for neural network
        input_blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), 
            swapRB=True, crop=False
        )
        
        self.neural_net.setInput(input_blob)
        network_outputs = self.neural_net.forward(
            self.neural_net.getUnconnectedOutLayersNames()
        )
        
        # Extract detection data
        detection_boxes = []
        confidence_scores = []
        object_classes = []
        
        for output_layer in network_outputs:
            for detection in output_layer:
                class_scores = detection[5:]
                detected_class = np.argmax(class_scores)
                confidence = class_scores[detected_class]
                
                if confidence > self.detection_confidence:
                    # Calculate bounding box coordinates
                    box_center_x = int(detection[0] * img_width)
                    box_center_y = int(detection[1] * img_height)
                    box_width = int(detection[2] * img_width)
                    box_height = int(detection[3] * img_height)
                    
                    box_x = int(box_center_x - box_width/2)
                    box_y = int(box_center_y - box_height/2)
                    
                    detection_boxes.append([box_x, box_y, box_width, box_height])
                    confidence_scores.append(float(confidence))
                    object_classes.append(detected_class)
        
        # Remove overlapping detections
        final_indices = cv2.dnn.NMSBoxes(
            detection_boxes, confidence_scores, 
            self.detection_confidence, self.overlap_threshold
        )
        
        return detection_boxes, confidence_scores, object_classes, final_indices
    
    def calculate_object_distance(self, object_type, pixel_height):
        """Estimate distance using object size"""
        known_size = self.item_dimensions.get(object_type, 0.4)
        camera_focal = 750  # Calibrated focal length
        
        if pixel_height > 0:
            estimated_distance = (known_size * camera_focal) / pixel_height
            return max(0.3, min(estimated_distance, 30.0))
        return 0
    
    def announce_detection(self, found_objects):
        """Provide voice feedback for detections"""
        current_timestamp = time.time()
        
        if (not found_objects or not self.voice_engine or 
            current_timestamp - self.last_voice_time < self.voice_delay):
            return
        
        # Select highest confidence detection
        primary_object = max(found_objects, key=lambda obj: obj['confidence'])
        
        announcement = f"{primary_object['type']} found at {primary_object['distance']:.1f} meters"
        
        try:
            self.voice_engine.say(announcement)
            self.voice_engine.runAndWait()
            self.last_voice_time = current_timestamp
        except Exception:
            pass
    
    def render_detections(self, image, boxes, scores, classes, indices):
        """Draw detection results on image"""
        detected_items = []
        
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                object_name = self.object_labels[classes[idx]]
                confidence = scores[idx]
                distance = self.calculate_object_distance(object_name, h)
                
                detected_items.append({
                    'type': object_name,
                    'confidence': confidence,
                    'distance': distance
                })
                
                # Draw bounding rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add text label
                label_text = f'{object_name}: {confidence:.2f} ({distance:.1f}m)'
                cv2.putText(image, label_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detected_items
    
    def execute_detection_loop(self):
        """Main processing loop"""
        if not self.setup_system():
            return False
        
        print("AccessVision active - Press 'q' to exit")
        
        try:
            while True:
                success, current_frame = self.video_capture.read()
                if not success:
                    print("Frame capture failed")
                    break
                
                # Perform object detection
                boxes, scores, classes, indices = self.analyze_frame(current_frame)
                
                # Render results and get detections
                found_objects = self.render_detections(
                    current_frame, boxes, scores, classes, indices
                )
                
                # Provide voice feedback
                self.announce_detection(found_objects)
                
                # Show processed frame
                cv2.imshow('AccessVision', current_frame)
                
                # Check for exit command
                pressed_key = cv2.waitKey(1) & 0xFF
                if pressed_key == ord('q') or pressed_key == 27:
                    break
                    
        except KeyboardInterrupt:
            print("System stopped by user")
        except Exception as error:
            print(f"Runtime error: {error}")
        finally:
            self.cleanup_resources()
        
        return True
    
    def cleanup_resources(self):
        """Release all system resources"""
        if self.video_capture:
            self.video_capture.release()
        
        cv2.destroyAllWindows()
        
        if self.voice_engine:
            try:
                self.voice_engine.stop()
            except:
                pass
        
        print("Resources cleaned up")

def launch_application():
    """Application entry point"""
    vision_system = AccessVision()
    
    try:
        vision_system.execute_detection_loop()
    except Exception as error:
        print(f"Application error: {error}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = launch_application()
    input("Press Enter to close...")
    sys.exit(exit_code)