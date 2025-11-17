# OPENCV — Road Lane Detection

An advanced **road lane detection system** built using OpenCV.  
The project processes images or videos, detects lane boundaries, highlights the drivable lane area, and calculates the vehicle’s lateral offset.

---

## Features

- Works fully on local machine (no Colab dependencies)
- Lane boundary and lane-area highlighting
- Accurate sliding-window lane pixel detection
- Polynomial curve fitting for lane lines
- Bird’s-eye view transformation (perspective warp)
- Vehicle center offset calculation (in meters)
- Smooth lane overlay on output video

---

## Pipeline Overview

1. **Camera Calibration (Optional)**  
   - Uses identity camera matrix and zero distortion by default.

2. **Binary Thresholding** (`to_binary(img)`)  
   - Combines HLS S-channel and grayscale thresholding to isolate lane pixels.

3. **Perspective Transform** (`warp_perspective(img)`)  
   - Converts road image to a top-down view for better lane detection.

4. **Sliding Windows Lane Detection** (`find_lane_pixels(binary)`)  
   - Uses histogram-based peak detection and sliding windows to track left/right lane lines.

5. **Polynomial Fitting**  
   - Fits each lane line with a 2nd-order polynomial.

6. **Lane Visualization** (`draw_lane()`)  
   - Creates a filled polygon between lane boundaries and warps it back to original perspective.

7. **Vehicle Offset Calculation** (`compute_offset()`)  
   - Computes lateral shift of the vehicle from the lane center.

8. **Video Processing** (`process_video()`)  
   - Reads input video, applies full pipeline, and generates processed output.

---

## Usage

### Process a video:
```bash
python lane_detection.py --video input.mp4 --output output.mp4
