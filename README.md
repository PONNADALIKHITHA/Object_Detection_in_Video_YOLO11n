# Object Detection in Video using YOLO11n

**Student:** Ponnada Likhitha  
**Roll No.:** 23341A1293  
**Assigned Project:** Project 32 — Object Detection in Video  
**Assigned Pre-trained Model:** YOLO11n

## Assignment alignment

This project follows the attached "PRE-TRAINED DEEP LEARNING MODELS — Individual Mini Project Assignment" structure:

1. Project Title
2. Student Details
3. Problem Statement
4. Objective
5. Introduction to the Pre-trained Model
6. Why This Model Was Selected
7. Model Architecture / Working Principle
8. Libraries and Requirements
9. Installation
10. Importing Libraries
11. Loading the Pre-trained Model
12. Input Preparation
13. Pre-processing
14. Model Inference
15. Output Processing
16. Result Visualization
17. Test Cases
18. Results / Observations
19. Limitations
20. Future Scope
21. Conclusion

The implementation uses a pre-trained YOLO11n model for inference only. It does not train a deep-learning model from scratch.

## 1. Environment

Recommended:
- Python 3.9+
- VS Code or Jupyter Notebook
- Windows/Linux/macOS

## 2. Installation

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

## 3. Input videos

Place three recorded videos in `input_videos/`:

```text
input_videos/
├── test_case_1.mp4
├── test_case_2.mp4
└── test_case_3.mp4
```

The videos should contain real objects suitable for general YOLO object detection. The assignment requires at least 3 different inputs/test cases.

## 4. Run the project

### Process all three test cases

```bash
python object_detection_video.py
```

### Process one specific video

```bash
python object_detection_video.py --input input_videos/test_case_1.mp4 --output outputs/test_case_1.mp4
```

### Show the live detection window

```bash
python object_detection_video.py --input input_videos/test_case_1.mp4 --show
```

Press `Q` to stop the display.

The first run may automatically download the pre-trained `yolo11n.pt` weights through Ultralytics.

## 5. Output

Processed videos are written to `outputs/`.

The program also writes a detection summary to `results/detection_summary.csv`.

Download the processed sample videos:

- [test_case_1_detected.mp4](outputs/test_case_1_detected.mp4)
- [test_case_2_detected.mp4](outputs/test_case_2_detected.mp4)
- [test_case_3_detected.mp4](outputs/test_case_3_detected.mp4)

## 6. Notebook

Open:

```text
object_detection_video.ipynb
```

The notebook follows the assignment's requested submission structure and contains explanations, installation, imports, model loading, input preparation, preprocessing, inference, output processing, visualization, three test cases, results/observations, limitations, future scope, conclusion, and viva preparation.

## 7. Important submission note

This ZIP does not contain fabricated detection screenshots, fabricated accuracy values, or fabricated test results. Add your three actual videos, run the project, and use the generated outputs/screenshots as the demonstration evidence.

## 8. Model source and license

Model source: Ultralytics YOLO11 documentation/repository.

YOLO11 models are provided under AGPL-3.0 and Enterprise licenses. Check the current official licensing terms before redistribution or commercial use.

Official documentation:
https://docs.ultralytics.com/models/yolo11

Official repository:
https://github.com/ultralytics/ultralytics
