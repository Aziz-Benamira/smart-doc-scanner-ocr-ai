import os
import uuid
import cv2
import numpy as np
import imutils
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_local
from skimage.feature import canny
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash



sigma_list = [(1,2), (0.5,2.2), (0.5,2.5), (1.5, 2)]
# --- Helper Functions (Copy order_points and four_point_transform here) ---
def order_points(pts):
    # Convert to list for easier manipulation
    pts = np.array(pts, dtype="float32").tolist()
    
    # Bottom-left: smallest x, then largest y
    bl = sorted(pts, key=lambda p: (p[0], -p[1]))[0]  # -p[1] for largest y
    remaining = [p for p in pts if p != bl]
    
    # Top-left: smallest y, then smallest x
    tl = sorted(remaining, key=lambda p: (p[1], p[0]))[0]
    remaining = [p for p in remaining if p != tl]
    
    # Top-right: smallest y, then largest x
    tr = sorted(remaining, key=lambda p: (p[1], -p[0]))[0]
    remaining = [p for p in remaining if p != tr]
    
    # Bottom-right: remaining point
    br = remaining[0]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    print(dst)
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- Core Processing Function (Copy process_document here) ---
def process_document(image, selected_filter="adaptive_threshold" , sigma_1 = 1, sigma_2 = 2):
    # ... (copy function code from above) ...
    orig = image.copy()
    ratio = 1/(image.shape[0] / 600.0)
    image_resized = rescale(image,ratio,channel_axis=2)
    image_resized_uint8 = np.uint8(image_resized * 255)
    gray = rgb2gray(image_resized)
    # Couple de sigma
    blurred_skimage = gaussian(gray, sigma=sigma_1)
    edged = canny(blurred_skimage, sigma=sigma_2)
    
    
    edged_uint8 = np.uint8(edged * 255)
    cnts = cv2.findContours(edged_uint8.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.countNonZero(cv2.drawContours(np.zeros_like(edged_uint8), [c], -1, (255), thickness=cv2.FILLED)), reverse=True)[:10] 
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            screenCnt=screenCnt.reshape(4,-1)
            break
    else:
        raise ValueError("Document contour not found.")
    screenCnt_original = screenCnt / ratio
    
    
    warped = four_point_transform(orig,screenCnt_original )
    
    
    processed_output = None
    if selected_filter == "adaptive_threshold":
        # Ensure warped isn't empty before processing
        if warped is None or warped.size == 0:
             return None, None, "Perspective transform resulted in an empty image."
        try:
            warped_gray = rgb2gray(warped)
            block_size = 15
            local_thresh = threshold_local(warped_gray, block_size, offset=0.1)
            processed_output = ((warped_gray > local_thresh)*255).astype("uint8")
        except cv2.error as e:
            return None, None, f"OpenCV error during thresholding: {e}"

    elif selected_filter == "grayscale":
        if warped is None or warped.size == 0:
             return None, None, "Perspective transform resulted in an empty image."
        processed_output = rgb2gray(warped)
      

    elif selected_filter == "color":
        processed_output = warped

    else: # Default to color
         processed_output = warped

    # Resize original for display - use the initially loaded 'orig'


    # Final check on processed output
    if processed_output is None or processed_output.size == 0:
        return None, None, "Processing failed to produce an output image."

    return processed_output, None


# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key' # Change this!
# So that  app works even if ran from outside folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'processed')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # --- File Handling ---
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # --- Processing ---
            try:
                # Generate unique filenames
                original_filename_base = str(uuid.uuid4())
                original_extension = file.filename.rsplit('.', 1)[1].lower()
                original_filename = f"{original_filename_base}.{original_extension}"
                processed_filename = f"{original_filename_base}_processed.jpg" # Save processed as JPG

                original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

                file.save(original_path)

                # Read image with OpenCV
                image_cv = io.imread(original_path)
                if image_cv is None:
                    flash('Could not read uploaded image file. It might be corrupted or in an unsupported format.', 'error')
                    # Clean up saved file if reading failed
                    if os.path.exists(original_path):
                        os.remove(original_path)
                    return redirect(request.url)


                selected_filter = request.form.get('filter', 'adaptive_threshold') # Get selected filter

                # Process the document
                processed_image, error_msg = process_document(image_cv, selected_filter)
                
                for sigma_1, sigma_2 in sigma_list:
                    processed_image, error_msg = process_document(image_cv, selected_filter, sigma_1, sigma_2)
                    if not error_msg:
                        break

                if error_msg:
                    flash(f'Processing Error: {error_msg}', 'error')
                    # Clean up original file if processing failed early
                    if os.path.exists(original_path):
                         os.remove(original_path)
                    return redirect(request.url)

                if processed_image is None:
                     flash('Processing failed to return an image.', 'error')
                     if os.path.exists(original_path):
                         os.remove(original_path)
                     return redirect(request.url)

                # Save the processed image (use JPG for broad compatibility)
                plt.imsave(processed_path, processed_image, cmap='gray' if selected_filter != 'color' else None, format='png')

                flash('Image processed successfully!', 'success')

                # Render template with results
                return render_template('index.html',
                                       original_filename=original_filename,
                                       processed_filename=processed_filename,
                                       selected_filter=selected_filter)

            except Exception as e:
                flash(f'An unexpected error occurred: {str(e)}', 'error')
                # Attempt cleanup on error
                if 'original_path' in locals() and os.path.exists(original_path):
                     os.remove(original_path)
                if 'processed_path' in locals() and os.path.exists(processed_path):
                     os.remove(processed_path)
                return redirect(request.url) # Redirect back to upload form on error

        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg, webp', 'error')
            return redirect(request.url)

    # --- GET Request ---
    # If it's a GET request or the POST failed before processing, just show the upload form
    return render_template('index.html')

# --- Routes to Serve Images ---
@app.route('/uploads/<filename>')
def serve_original_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# --- Route to Download Processed Image ---
@app.route('/download/<filename>')
def download_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production