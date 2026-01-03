# AccessVision - Assistive Technology System

## Overview
AccessVision is an original assistive technology system designed to help visually impaired users navigate their environment through real-time object detection and voice announcements.

## Features
- Real-time object detection using computer vision
- Voice announcements of detected objects
- Distance estimation for spatial awareness
- Clean, accessible interface
- Proper resource management

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model Files
```bash
python download_models.py
```

### 3. Run the Application
```bash
python access_vision.py
```

## Controls
- Press 'q' or ESC to exit
- System announces detected objects every 4 seconds

## Requirements
- Python 3.7+
- Webcam
- Speakers/headphones

## File Structure
```
AccessVision/
├── access_vision.py      # Main application
├── coco.names           # Object class names
├── download_models.py   # Model downloader
├── requirements.txt     # Dependencies
├── models/             # Model files (created by downloader)
│   ├── yolov3.weights  # YOLO weights
│   └── yolov3.cfg      # YOLO config
└── README.md           # This file
```

## Technical Details
- Uses YOLOv3 for object detection
- Implements distance estimation based on object size
- Non-blocking voice synthesis
- Automatic resource cleanup

## Original Implementation
This is a completely original implementation created specifically for accessibility purposes. All code has been written from scratch with unique architecture and implementation patterns.

## License
MIT License - See LICENSE file for details