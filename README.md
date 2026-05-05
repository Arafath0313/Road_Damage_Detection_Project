# Video Processing Pipeline

A comprehensive Python-based image and video processing pipeline that extracts frames from video, applies various image enhancement and edge detection techniques, performs morphological operations, and detects contours.

[Video Link](https://drive.google.com/file/d/1x9WgAh5QEggWYBFb5mEHz223sd7lXxjS/view?usp=drive_link)

## Project Overview

This project implements a multi-stage image processing pipeline designed to process video frames through a series of enhancement and analysis steps:

1. **Frame Extraction** - Extract individual frames from video files
2. **Grayscale Conversion** - Convert frames to grayscale
3. **Denoising** - Remove noise from frames
4. **Contrast Enhancement** - Improve image contrast
5. **Canny Edge Detection** - Detect edges in frames
6. **Morphological Operations** - Apply erosion/dilation operations
7. **Contour Detection** - Identify and extract contours
8. **Overlay & Visualization** - Create combined visualization outputs

## Directory Structure

```
.
├── Dataset/
│   ├── 01_Extracted_original_frames/
│   ├── 02_Grayscale_Frames/
│   ├── 03_Partially_processed_frames/
│   ├── 04_Final_enhanced_frames/
│   ├── 05_Canny_Frames/
│   ├── 06_Morph_Frames/
│   ├── 07_Segmentation_outputs/
│   └── 08_Final_marked_outputs/
├── Notebooks/
│   ├── Main_Algorithm.ipynb
│   ├── Extract_Function.ipynb
│   ├── Gray_Convert_Function.ipynb
│   ├── Denoise_Function.ipynb
│   ├── Contrast_Function.ipynb
│   ├── Canny_Function.ipynb
│   ├── Morphology_Function.ipynb
│   ├── Contour_Function.ipynb
│   ├── Final_Overlay_Function.ipynb
│   └── Show_Combined_Frames.ipynb
├── Raw_Video/
└── README.md
```

## Notebooks

### Main Notebooks

- **Main_Algorithm.ipynb** - Master notebook that orchestrates the entire pipeline by running all function notebooks in sequence
- **Extract_Function.ipynb** - Extracts frames from video input
- **Gray_Convert_Function.ipynb** - Converts frames to grayscale
- **Denoise_Function.ipynb** - Applies denoising filters
- **Contrast_Function.ipynb** - Enhances image contrast
- **Canny_Function.ipynb** - Applies Canny edge detection
- **Morphology_Function.ipynb** - Performs morphological operations (erosion, dilation)
- **Contour_Function.ipynb** - Detects and filters contours by area
- **Final_Overlay_Function.ipynb** - Overlays contours on original frames
- **Show_Combined_Frames.ipynb** - Displays and compares processed frames

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

```bash
pip install opencv-python numpy matplotlib jupyter
```

## Usage

### Running the Full Pipeline

Open and run **Main_Algorithm.ipynb** to execute the complete processing pipeline. The notebook will:

1. Import all required libraries
2. Load all function modules via `%run` commands
3. Execute each processing step in sequence
4. Generate processed frames in respective dataset folders

### Running Individual Steps

Each processing step can be run independently by opening its corresponding notebook:

```python
# Example: Extract frames only
video_path = r"C:\path\to\video.mov"
extract_function(video_path)

# Example: Apply contrast adjustment
input_folder = r"C:\path\to\grayscale_frames"
contrast_function(input_folder)

# Example: Detect contours with minimum area threshold
contour_function(input_folder, min_area=300)
```

## Pipeline Parameters

Key parameters that can be customized:

- **min_area** - Minimum contour area (default: 200-300 pixels) - filters small contours
- **video_path** - Path to input video file
- **input_folder** - Path to input image frames

## Output

The pipeline generates output images at each stage:

1. Extracted frames
2. Grayscale frames
3. Denoised frames
4. Contrast-enhanced frames
5. Canny edge detection results
6. Morphologically processed frames
7. Contour detection results
8. Final overlaid frames with marked contours

## Dependencies

- **OpenCV (cv2)** - Image processing and video handling
- **NumPy** - Numerical array operations
- **Matplotlib** - Visualization and frame display

## Notes

- Video file format: MOV, MP4, AVI (OpenCV supported formats)
- All frames are processed sequentially through the pipeline
- Output folders are created automatically if they don't exist
- Adjust min_area parameter in contour detection based on object size requirements

## License

This project is for educational and research purposes.

## Author

Created as part of a computer vision project involving video frame analysis and contour detection.
