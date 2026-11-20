# Virtual Drawing Board

A Python-based virtual drawing board that uses real-time hand tracking and gesture recognition to allow users to draw and manipulate shapes through a webcam.

## Features
- Real-time hand gesture recognition
- Freehand drawing using pinch gestures
- Shape creation (circles, rectangles, and triangles)
- Shape movement and resizing
- Undo, redo, and clear functions
- Full-screen drawing interface

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy

## Installation

Python 3.11 is recommended.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python virtual_board.py
```

Make sure your webcam is connected and accessible.

## Controls

- **M** – Toggle shape mode
- **U** – Undo
- **R** – Redo
- **C** – Clear the board
- **Q** – Quit the application
