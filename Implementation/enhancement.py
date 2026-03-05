import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



#  EXTRACT FRAMES FUNCTION
def extract_function(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

   
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25
    frame_interval = max(1, int(round(fps)))  

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames.")
    return output_folder




#  GRAYSCALE FUNCTION
def gray_convert_function(input_folder, output_folder="gray_frames"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(input_folder, filename))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output_folder, filename), gray)

    print("Grayscale conversion completed.")
    return output_folder




#  DENOISING FUNCTION
def denoise_function(input_folder, output_folder="denoised_frames"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(input_folder, filename), 0)
            denoised = cv2.GaussianBlur(img, (5,5), 0)
            cv2.imwrite(os.path.join(output_folder, filename), denoised)

    print("Noise reduction completed.")
    return output_folder
    


#  CONTRAST ENHANCEMENT (CLAHE)
def contrast_function(input_folder, output_folder="contrast_frames"):
    os.makedirs(output_folder, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(input_folder, filename), 0)
            enhanced = clahe.apply(img)
            cv2.imwrite(os.path.join(output_folder, filename), enhanced)

    print("Contrast enhancement completed.")
    return output_folder



#  SHARPEN FUNCTION
def sharpen_function(input_folder, output_folder="sharpened_frames"):
    os.makedirs(output_folder, exist_ok=True)

    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(input_folder, filename), 0)
            sharpened = cv2.filter2D(img, -1, kernel)
            cv2.imwrite(os.path.join(output_folder, filename), sharpened)

    print("Sharpening completed.")
    return output_folder


# Calculate PSNR
def calculate_psnr(original_img, processed_img):
    original = original_img.astype(np.float32)
    processed = processed_img.astype(np.float32)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


# Display function
def display_function(frame_name,
                     original_path,
                     gray_path,
                     contrast_path,
                     sharpened_path):
    
    original = cv2.imread(os.path.join(original_path, frame_name))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)


    original_bgr = cv2.imread(os.path.join(original_path, frame_name))
    original_ref = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.imread(os.path.join(gray_path, frame_name), 0)
    contrast = cv2.imread(os.path.join(contrast_path, frame_name), 0)
    sharpened = cv2.imread(os.path.join(sharpened_path, frame_name), 0)

    # Calculate PSNR
    psnr_gray = calculate_psnr(original_ref, gray)
    psnr_contrast = calculate_psnr(original_ref, contrast)
    psnr_sharp = calculate_psnr(original_ref, sharpened)

    print(f"PSNR (Original vs Grayscale): {psnr_gray:.2f} dB")
    print(f"PSNR (Original vs CLAHE): {psnr_contrast:.2f} dB")
    print(f"PSNR (Original vs Sharpened): {psnr_sharp:.2f} dB")

    
    plt.figure(figsize=(14,8))

    plt.subplot(2,2,1)
    plt.imshow(original_rgb, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(gray, cmap='gray')
    plt.title(f"Grayscale\nPSNR: {psnr_gray:.2f} dB")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(contrast, cmap='gray')
    plt.title(f"CLAHE\nPSNR: {psnr_contrast:.2f} dB")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(sharpened, cmap='gray')
    plt.title(f"Sharpened\nPSNR: {psnr_sharp:.2f} dB")
    plt.axis("off")

    print("Are original_ref and gray identical?", np.array_equal(original_ref, gray))
    plt.tight_layout()
    plt.show()




#  FUNCTION CALLS (PIPELINE)
video_file = "one.mov"

frames_folder = extract_function(video_file)
gray_folder = gray_convert_function(frames_folder)
denoise_folder = denoise_function(gray_folder)
contrast_folder = contrast_function(denoise_folder)
sharpen_folder = sharpen_function(contrast_folder)

display_function("frame_0004.png",
                 frames_folder,
                 gray_folder,
                 contrast_folder,
                 sharpen_folder)
