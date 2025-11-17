"""
scripts/lane_detection.py

Modified to run in VS Code / local terminal and also usable in Colab.

Usage (local / VS Code):
    python scripts/lane_detection.py --video input/your_video.mp4 --output output/lane_output.mp4

Usage (Colab):
    # either run with --video after uploading file to runtime, or
    python scripts/lane_detection.py
    # without --video inside Colab the upload widget will appear
"""
import os
import cv2
import numpy as np
import argparse

# try to detect google.colab environment
def _is_colab():
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

_COLAB = _is_colab()
if _COLAB:
    from google.colab import files  # type: ignore

# ===============================
# 1. CAMERA CALIBRATION (optional)
# ===============================
camera_matrix = np.eye(3)
dist_coeffs = np.zeros((5,))


# ===============================
# 2. HELPER FUNCTIONS
# ===============================
def undistort(img):
    return cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)


def to_binary(img, s_thresh=150, g_thresh=180):
    """
    HLS S-channel + Grayscale thresholding -> combined binary mask (0/255)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]
    _, s_bin = cv2.threshold(s, s_thresh, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, g_bin = cv2.threshold(gray, g_thresh, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(s_bin, g_bin)
    return combined


def warp_perspective(img):
    """
    Perspective warp - returns (warped, Minv)
    Points are fractions of width/height and may need tuning per camera.
    """
    h, w = img.shape[:2]
    src = np.float32([
        [w * 0.45, h * 0.63],
        [w * 0.10, h * 0.95],
        [w * 0.90, h * 0.95],
        [w * 0.55, h * 0.63]
    ])
    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.20, h],
        [w * 0.80, h],
        [w * 0.80, 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, Minv


def find_lane_pixels(binary):
    """
    Sliding-window search to return leftx,lefty,rightx,righty arrays (may be empty).
    """
    histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    if histogram.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint]) if midpoint > 0 else 0
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint if midpoint > 0 else 0

    nwindows = 9
    window_height = np.int32(binary.shape[0] // nwindows)

    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 80
    minpix = 40

    left_lane = []
    right_lane = []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane.append(good_left)
        right_lane.append(good_right)

        if len(good_left) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right]))

    if len(left_lane) == 0 or len(right_lane) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    # guard for empty indices
    if left_lane.size == 0 or right_lane.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    leftx = nonzerox[left_lane]
    lefty = nonzeroy[left_lane]
    rightx = nonzerox[right_lane]
    righty = nonzeroy[right_lane]

    return leftx, lefty, rightx, righty


def safe_polyfit(y, x, order=2):
    """
    Return polynomial coefficients or None if not enough points.
    Fit polynomial x = f(y).
    """
    if x.size < (order + 1) or y.size < (order + 1):
        return None
    try:
        coeffs = np.polyfit(y, x, order)
        return coeffs
    except Exception:
        return None


def draw_lane(original, warped, Minv, left_fit, right_fit):
    """
    Draw lane polygon from left/right polynomial fits and overlay onto original frame.
    """
    h, w = original.shape[:2]
    ploty = np.linspace(0, h - 1, h)

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)
    return result


def compute_offset(left_fit, right_fit, h, w):
    """
    Compute lateral offset from lane center in meters.
    Positive offset means vehicle is left of lane center (needs steer right).
    """
    left_x = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    right_x = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
    lane_center = (left_x + right_x) / 2.0
    vehicle_center = w / 2.0
    xm_per_pix = 3.7 / 700.0
    offset = (vehicle_center - lane_center) * xm_per_pix
    return offset


# ===============================
# 3. PROCESS VIDEO
# ===============================
def process_video(input_path, output_path="lane_output.mp4", verbose=False, colab_download=True):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        # fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"M", "J", "P", "G")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if verbose:
        print(f"Processing {input_path} -> {output_path} (size={w}x{h}, fps={fps:.2f})")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        und = undistort(frame)
        binary = to_binary(und)
        warped, Minv = warp_perspective(binary)

        leftx, lefty, rightx, righty = find_lane_pixels(warped)
        if leftx.size == 0 or rightx.size == 0:
            if verbose:
                print(f"Frame {frame_idx}: lanes not found, writing original frame")
            writer.write(frame)
            continue

        left_fit = safe_polyfit(lefty, leftx, order=2)
        right_fit = safe_polyfit(righty, rightx, order=2)
        if left_fit is None or right_fit is None:
            if verbose:
                print(f"Frame {frame_idx}: insufficient points for polyfit")
            writer.write(frame)
            continue

        lane_img = draw_lane(und, warped, Minv, left_fit, right_fit)
        offset = compute_offset(left_fit, right_fit, h, w)

        cv2.putText(lane_img, f"Vehicle Offset: {offset:+.2f} m", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lane_img, "Good Lane Keeping", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(lane_img)

    cap.release()
    writer.release()

    if verbose:
        print(f"Processing finished. Output saved to: {output_path}")

    # trigger download in Colab if requested
    if _COLAB and colab_download:
        try:
            from google.colab import files as colab_files  # type: ignore
            colab_files.download(output_path)
        except Exception:
            pass


# ===============================
# 4. ENTRYPOINT (CLI + Colab upload)
# ===============================
def colab_upload_prompt():
    if not _COLAB:
        return None
    uploaded = files.upload()
    if not uploaded:
        return None
    return list(uploaded.keys())[0]


def main():
    parser = argparse.ArgumentParser(description="Advanced Lane Detection (local & Colab)")
    parser.add_argument("--video", "-v", help="Input video path (local). If omitted in Colab, upload widget will appear.", required=False)
    parser.add_argument("--output", "-o", default="lane_output.mp4", help="Output video path")
    parser.add_argument("--no-colab-download", action="store_true", help="Do not auto-download output in Colab")
    parser.add_argument("--verbose", action="store_true", help="Verbose prints")
    args = parser.parse_args()

    input_path = args.video
    if input_path is None:
        if _COLAB:
            print("No --video provided. Opening Colab upload widget...")
            uploaded = colab_upload_prompt()
            if uploaded is None:
                print("No file uploaded. Exiting.")
                return
            input_path = uploaded
        else:
            print("Error: --video is required when running locally (VS Code / terminal).")
            print("Example: python scripts/lane_detection.py --video input/your_video.mp4 --output output/out.mp4")
            return

    # ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    process_video(input_path, output_path=args.output, verbose=args.verbose, colab_download=not args.no_colab_download)


if __name__ == "__main__":
    main()
