# Smart Cities Object Detection & Tracking Microservice

A production-ready microservice for real-time multi-object detection and tracking in smart city infrastructures. This containerized service processes video streams from multiple cameras simultaneously, providing scalable tracking capabilities for urban monitoring, traffic management, and public safety applications.

## Overview

This microservice is designed for smart city ecosystems where hundreds or thousands of cameras need centralized processing. It combines [YOLOv8](https://github.com/ultralytics/ultralytics) detection with state-of-the-art tracking algorithms, offering horizontal scalability, load balancing, and enterprise-grade reliability.

### Supported Tracking Algorithms

- **[DeepOCSORT](https://arxiv.org/abs/2302.11813)** - Enhanced OCSORT with deep appearance features
- **[StrongSORT](https://arxiv.org/abs/2202.13514)** - Robust multi-object tracking with appearance modeling
- **[BoTSORT](https://arxiv.org/abs/2206.14651)** - Combining motion and appearance cues
- **[OCSORT](https://arxiv.org/abs/2203.14360)** - Observation-centric SORT
- **[ByteTrack](https://arxiv.org/abs/2110.06864)** - High-performance multi-object tracking

### Smart Cities Features

- **Multi-Camera Processing**: Handle hundreds of simultaneous camera feeds
- **Microservice Architecture**: Container-native design for cloud-native deployments
- **Horizontal Scaling**: Auto-scaling based on camera load and processing demand
- **Real-Time Analytics**: Live object detection, tracking, and behavioral analysis
- **Edge-to-Cloud**: Support for edge processing with cloud aggregation
- **High Availability**: Fault-tolerant design with automatic failover
- **API-First**: RESTful and streaming APIs for easy integration
- **Multi-Modal Support**: Object detection, segmentation, and pose estimation

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch >= 1.7.0

### Installation

```bash
# Clone repository
git clone https://github.com/HASHIRAMA21/yolov8_tracking.git
cd yolov8_tracking

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
# Models will be auto-downloaded on first use
```

### Deployment Options

#### Docker Deployment
```bash
# Build container
docker build -t yolov8-tracking .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace yolov8-tracking
```

#### Production Environment
For production deployments, consider:
- Model optimization (TensorRT, ONNX)
- Batch processing for higher throughput
- Custom ReID models for domain-specific tracking

### Custom Detection Models

The framework supports any detection model that outputs:
```
Nx6 format: [x1, y1, x2, y2, confidence, class]
```

## Usage

### Basic Tracking

```bash
# Object detection with tracking
python track.py --source video.mp4 --yolo-model yolov8n.pt --tracking-method deepocsort

# Segmentation with tracking
python track.py --source video.mp4 --yolo-model yolov8n-seg.pt --tracking-method strongsort

# Pose estimation with tracking
python track.py --source video.mp4 --yolo-model yolov8n-pose.pt --tracking-method botsort
```

### Input Sources

```bash
# Webcam
python track.py --source 0

# Video file
python track.py --source path/to/video.mp4

# Image directory
python track.py --source path/to/images/

# RTSP stream
python track.py --source "rtsp://example.com/stream"

# YouTube video
python track.py --source "https://youtu.be/VIDEO_ID"
```

### Configuration Options

#### Model Selection
```bash
# Performance vs Speed trade-off
python track.py --yolo-model yolov8n.pt    # Fastest, lower accuracy
python track.py --yolo-model yolov8s.pt    # Balanced
python track.py --yolo-model yolov8m.pt    # Better accuracy
python track.py --yolo-model yolov8l.pt    # High accuracy
python track.py --yolo-model yolov8x.pt    # Best accuracy, slower
```

#### Tracking Algorithms
```bash
python track.py --tracking-method deepocsort   # Recommended for most use cases
python track.py --tracking-method strongsort   # Best for crowded scenes
python track.py --tracking-method ocsort       # Fast, good for real-time
python track.py --tracking-method bytetrack    # High-performance baseline
python track.py --tracking-method botsort      # Robust motion modeling
```

#### Class Filtering
```bash
# Track specific classes only (COCO dataset)
python track.py --classes 0      # Person only
python track.py --classes 0 2    # Person and car
python track.py --classes 16 17  # Cat and dog
```

### Advanced Configuration

#### ReID Model Selection
```bash
# Lightweight ReID model
python track.py --reid-weights osnet_x0_25_market1501.pt

# High-accuracy ReID model
python track.py --reid-weights osnet_x1_0_msmt17.pt
```

#### Output Options
```bash
# Save MOT-compliant results
python track.py --save-txt --project runs/tracking --name experiment_1

# Display real-time results
python track.py --show

# Save video output
python track.py --save
```

### Hyperparameter Optimization

Automatically optimize tracker parameters for your dataset:

```bash
# Optimize for MOT17 dataset
python evolve.py --tracking-method strongsort --benchmark MOT17 --n-trials 100

# Optimize for custom dataset
python evolve.py --tracking-method deepocsort --benchmark /path/to/custom/data --objective HOTA
```

### Performance Tuning

```bash
# GPU acceleration
python track.py --device 0                    # Use GPU 0
python track.py --device 0,1                  # Multi-GPU

# Inference optimization
python track.py --half                        # FP16 inference
python track.py --imgsz 640                   # Input resolution
python track.py --conf 0.5                    # Confidence threshold
```

## Model Export and Optimization

### Export to Production Formats

```bash
# Export YOLOv8 model
yolo export model=yolov8n.pt format=onnx       # ONNX export
yolo export model=yolov8n.pt format=engine     # TensorRT export
yolo export model=yolov8n.pt format=openvino   # OpenVINO export
```

### Supported Model Formats

| Format | Speed | Accuracy | Deployment |
|--------|-------|----------|------------|
| PyTorch (.pt) | Medium | High | Development |
| ONNX (.onnx) | Fast | High | Cross-platform |
| TensorRT (.engine) | Fastest | High | NVIDIA GPU |
| OpenVINO | Fast | High | Intel hardware |

## Project Structure

```
├── trackers/                 # Tracking algorithm implementations
│   ├── deepocsort/          # DeepOCSORT tracker
│   ├── strongsort/          # StrongSORT tracker
│   ├── botsort/             # BoTSORT tracker
│   ├── ocsort/              # OCSORT tracker
│   └── bytetrack/           # ByteTrack tracker
├── yolov8/                  # YOLOv8 detection models
├── weights/                 # Pre-trained model weights
├── track.py                 # Main tracking script
├── evolve.py                # Hyperparameter optimization
├── val.py                   # Evaluation script
└── requirements.txt         # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-tracker`)
3. Commit your changes (`git commit -am 'Add new tracker'`)
4. Push to the branch (`git push origin feature/new-tracker`)
5. Create a Pull Request

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{yolov8_tracking,
  title={YOLOv8 Multi-Object Tracking Framework},
  author={Your Name},
  year={2023},
  url={https://github.com/HASHIRAMA21/yolov8_tracking}
}
```

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [DeepOCSORT](https://arxiv.org/abs/2302.11813) authors
- [StrongSORT](https://arxiv.org/abs/2202.13514) authors
- [ByteTrack](https://arxiv.org/abs/2110.06864) authors

