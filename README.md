
 # Sign Language Detection System

A real-time gesture-recognition web application that uses computer vision to detect hand gestures via webcam and classify them as sign-language symbols.

## Overview

This project combines live hand-tracking with a trained classifier to recognize sign-language gestures in real time. The pipeline captures video from the browser, extracts hand-landmark features frame-by-frame, and feeds those features into a Scikit-learn model to predict the corresponding sign — all with low-latency inference through a lightweight HTML/CSS/JS front-end.

## Features

- Real-time webcam-based hand tracking
- Frame-by-frame landmark extraction (21 hand landmarks per detected hand)
- Gesture classification using a trained Scikit-learn model
- Low-latency inference suitable for live interaction
- Simple, dependency-light front-end (HTML/CSS/JS) for demoing predictions in the browser

## Tech Stack

| Layer | Technology |
|---|---|
| Hand tracking | [MediaPipe](https://developers.google.com/mediapipe) |
| Computer vision / video capture | [OpenCV](https://opencv.org/) |
| Classification model | [Scikit-learn](https://scikit-learn.org/) |
| Numerical processing | [NumPy](https://numpy.org/) |
| Language | Python |
| Front-end | HTML, CSS, JavaScript |

## How It Works

1. **Capture** — OpenCV reads live video frames from the webcam.
2. **Landmark detection** — MediaPipe's hand-tracking solution detects and extracts 21 3D landmarks per hand in each frame.
3. **Feature vector** — Landmark coordinates are normalized and converted into a fixed-length feature vector using NumPy.
4. **Classification** — The feature vector is passed to a trained Scikit-learn classifier, which predicts the corresponding gesture/sign.
5. **Display** — The predicted label is rendered back to the user through the web front-end in real time.

## Project Structure

```
sign-language-detection/
├── data/                  # <FILL IN: raw/processed gesture data, if included>
├── model/                 # Trained classifier artifact(s)
├── src/
│   ├── capture.py         # Webcam capture + frame handling
│   ├── landmarks.py       # MediaPipe hand-landmark extraction
│   ├── features.py        # Landmark → feature vector conversion
│   ├── train.py           # Model training script
│   └── predict.py         # Real-time inference loop
├── static/                # Front-end assets (CSS/JS)
├── templates/              # HTML templates
├── requirements.txt
└── README.md
```
> Adjust this structure to match your actual repo layout.

## Model Details

- **Algorithm:** <FILL IN: e.g. Random Forest / SVM / KNN classifier>
- **Input features:** Normalized hand-landmark coordinates (21 points × x/y/z)
- **Gesture classes:** <FILL IN: number and list of signs/letters recognized>
- **Training data:** <FILL IN: dataset source, number of samples per class, self-collected vs. public dataset>
- **Accuracy:** <FILL IN: only if you have a real evaluated number>

## Limitations

- Recognition accuracy depends on lighting conditions and hand visibility in frame
- Currently supports single-hand gestures <FILL IN: confirm if two-hand gestures are supported>
- Gesture set is limited to <FILL IN: static signs only / no continuous sign-language sentence recognition>

## Future Improvements

- Expand gesture vocabulary beyond the current class set
- Add support for continuous sign sequences (sentence-level recognition) rather than isolated gestures
- Improve robustness to varying lighting and background conditions
- Deploy as a hosted web app (e.g. Render/Railway) for live demoing
