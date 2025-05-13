import os
import uuid
import cv2
import numpy as np
import imutils
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# --- Helper Functions (Copy order_points and four_point_transform here) ---
def order_points(pts):
    # ... (copy function code from above) ...
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # ... (copy function code from above) ...
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
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
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- Core Processing Function (Copy process_document here) ---
def process_document(image_cv, selected_filter="adaptive_threshold"):
    # ... (copy function code from above) ...
    orig = image_cv.copy()
    ratio = image_cv.shape[0] / 500.0
    image_resized = imutils.resize(image_cv, height=500)

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
             # Extra check: Ensure the contour area is reasonably large
            if cv2.contourArea(approx) > (image_resized.shape[0] * image_resized.shape[1] * 0.05): # Reduced threshold slightly
               screenCnt = approx
               break

    if screenCnt is None:
        return None, None, "Could not find a 4-point document contour. Try a different image or adjust lighting."

    try:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    except Exception as e:
        # Catch potential errors during transformation (e.g., degenerate quad)
        return None, None, f"Error during perspective transform: {e}. Ensure points form a valid quadrilateral."


    processed_output = None
    if selected_filter == "adaptive_threshold":
        # Ensure warped isn't empty before processing
        if warped is None or warped.size == 0:
             return None, None, "Perspective transform resulted in an empty image."
        try:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            processed_output = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 10)
            processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR) # Keep 3 channels for consistency
        except cv2.error as e:
            return None, None, f"OpenCV error during thresholding: {e}"

    elif selected_filter == "grayscale":
        if warped is None or warped.size == 0:
             return None, None, "Perspective transform resulted in an empty image."
        processed_output = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR)

    elif selected_filter == "color":
        processed_output = warped

    else: # Default to color
         processed_output = warped

    # Resize original for display - use the initially loaded 'orig'
    orig_display = imutils.resize(orig, height=650)

    # Final check on processed output
    if processed_output is None or processed_output.size == 0:
        return None, None, "Processing failed to produce an output image."

    return processed_output, orig_display, None


# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key' # Change this!
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
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
                image_cv = cv2.imread(original_path)
                if image_cv is None:
                    flash('Could not read uploaded image file. It might be corrupted or in an unsupported format.', 'error')
                    # Clean up saved file if reading failed
                    if os.path.exists(original_path):
                        os.remove(original_path)
                    return redirect(request.url)


                selected_filter = request.form.get('filter', 'adaptive_threshold') # Get selected filter

                # Process the document
                processed_image, _, error_msg = process_document(image_cv, selected_filter) # We don't need orig_display here

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
                cv2.imwrite(processed_path, processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]) # Save with quality 90

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