# -*- coding: utf-8 -*-
import os
import uuid
import cv2
import numpy as np
import imutils
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pytesseract # For OCR
import replicate   # For Llama API via Replicate
from PIL import Image # Often needed by pytesseract

# --- Helper Functions (Image Processing) ---
def order_points(pts):
    """ Orders points: top-left, top-right, bottom-right, bottom-left """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """ Applies perspective transform to obtain bird's-eye view """
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

# --- Core Image Processing Function ---
def process_document(image_cv, selected_filter="adaptive_threshold"):
    """ Finds document contour, warps perspective, and applies filter """
    orig = image_cv.copy()
    if image_cv.size == 0: return None, None, "Input image is empty." # Basic check

    # Resize for faster contour detection
    ratio = image_cv.shape[0] / 500.0
    image_resized = imutils.resize(image_cv, height=500)
    if image_resized.size == 0: return None, None, "Resized image is empty."

    # Preprocessing for contour detection
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts: return None, None, "No contours found in image."
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Find the 4-point contour
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Basic area check to avoid tiny contours
            if cv2.contourArea(approx) > (image_resized.shape[0] * image_resized.shape[1] * 0.05):
               screenCnt = approx
               break

    if screenCnt is None:
        return None, None, "Could not find a suitable 4-point document contour."

    # Apply perspective transform to the original image
    try:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    except Exception as e:
        return None, None, f"Error during perspective transform: {e}."

    processed_output = None
    if warped is None or warped.size == 0:
        return None, None, "Perspective transform resulted in an empty image."

    # Apply selected filter to the warped image
    try:
        if selected_filter == "adaptive_threshold":
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            processed_output = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 10)
            processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR) # Keep 3 channels
        elif selected_filter == "grayscale":
            processed_output = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR)
        elif selected_filter == "color":
            processed_output = warped
        else:
            processed_output = warped # Default to color
    except cv2.error as e:
         return None, None, f"OpenCV error during filtering/thresholding: {e}"

    if processed_output is None or processed_output.size == 0:
        return None, None, "Processing failed to produce a final output image."

    # Return the processed image (orig_display isn't needed by backend logic)
    return processed_output, None, None

# --- Flask App Setup ---
app = Flask(__name__)
# IMPORTANT: Set a proper secret key for production or use environment variables
app.config['SECRET_KEY'] = 'some-very-random-and-unguessable-string-for-flask' # <-- CHANGE THIS!
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# --- Replicate API Token (Hardcoded) ---
# !! WARNING: Replace placeholder with your actual key. Only use hardcoding for strictly local development. !!
REPLICATE_API_TOKEN = "r8_SqA0KefWLrNR03dMK3HAMnCJ3luCMAb3MYMhS" # <--- PASTE YOUR REPLICATE TOKEN HERE

# --- Tesseract Configuration (Optional) ---
# If Python can't find Tesseract automatically, uncomment and set the correct path below.
# ---v--- Make sure this path points to your tesseract.exe (Win) or tesseract (Mac/Linux) ---v---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/opt/tesseract/bin/tesseract' # Example macOS
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Example Linux

# Create upload/processed directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- OCR Function ---
def perform_ocr(image_cv):
    """Performs OCR on an OpenCV image using Tesseract."""
    print("[DEBUG] Performing OCR...")
    if image_cv is None or image_cv.size == 0:
        print("[DEBUG] OCR skipped: Input image is empty.")
        return None, "Input image for OCR was empty."
    try:
        # Convert to grayscale if needed, Tesseract generally prefers grayscale/binary
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
             gray_for_ocr = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
             gray_for_ocr = image_cv

        # Optional: Apply thresholding for potentially better results on some images
        # _, gray_for_ocr = cv2.threshold(gray_for_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR
        text = pytesseract.image_to_string(gray_for_ocr, lang='eng') # Specify language if known
        print(f"[DEBUG] OCR Result Length: {len(text)}")
        return text, None
    except pytesseract.TesseractNotFoundError:
        error_msg = "Tesseract Error: Executable not found. Ensure Tesseract is installed and in PATH or configured in app.py."
        print(f"[ERROR] {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"OCR Error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return None, error_msg


# --- Llama Text Enhancement Function ---
def enhance_text_with_ai(text):
    """Uses Replicate API (Llama) to clean and format OCR text."""
    print("[DEBUG] Attempting AI enhancement...")
    # Check if the hardcoded token is valid (not empty or placeholder)
    if not REPLICATE_API_TOKEN or "..." in REPLICATE_API_TOKEN:
        err_msg = "AI client not configured (Replicate API token missing or placeholder in code)."
        print(f"[DEBUG] {err_msg}")
        return text, err_msg # Return original text and the error message

    if not text or not text.strip():
         print("[DEBUG] AI enhancement skipped: No input text.")
         return "", None # Return empty string if input is empty/whitespace

    # Set environment variable temporarily for the replicate client to pick up
    # This is belt-and-suspenders when hardcoding, but ensures the library finds it.
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
    print("[DEBUG] Set REPLICATE_API_TOKEN environment variable for this call.")

    # --- IMPORTANT: Select a Llama Model Identifier from Replicate ---
    # Go to replicate.com, find a Llama chat model (e.g., llama-2-13b-chat),
    # click API tab, and copy the identifier string (model_owner/model_name:version_hash)
    # Using a specific version hash is highly recommended for consistency!
    # Example (check replicate.com for the latest/correct one):
    model_identifier = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

    # Prepare the prompt specifically for Llama 2 Chat models
    prompt_text = f"""[INST] <<SYS>>
You are a helpful assistant that meticulously cleans and formats OCR text. Your primary goal is to correct errors and make the text highly readable and easily copy-pasteable, as if it were typed manually. Output *only* the cleaned text, without any conversational preamble, summaries, or explanations about your process.
<< /SYS >>

The following text was extracted from a document image using OCR and may contain various errors (misrecognized characters, incorrect spacing, formatting issues, fragmented sentences). Please correct these issues and format the text logically. Use standard paragraph breaks. If you detect lists or clear headings, attempt to format them appropriately (e.g., using '*' for list items).

Original OCR Text:
---
{text}
---

Cleaned and Formatted Text: [/INST]"""
    print(f"[DEBUG] Sending prompt to Replicate (first 100 chars): {prompt_text[:100]}...")

    try:
        print(f"[INFO] Calling Replicate API with model: {model_identifier}")
        # Call the Replicate API
        output_iterator = replicate.run(
            model_identifier,
            input={
                "prompt": prompt_text,
                "temperature": 0.2, # Lower temperature for more focused, less creative cleanup
                "max_new_tokens": 2500, # Adjust based on expected output length
                "top_p": 0.9, # Common sampling parameter
                # Add other relevant parameters for the specific model if needed
            }
        )

        # Process the output iterator to get the full response string
        enhanced_text_list = list(output_iterator)
        enhanced_text = "".join(enhanced_text_list)

        print(f"[DEBUG] Raw output list from Replicate: {enhanced_text_list}") # See individual chunks if needed
        print(f"[DEBUG] Joined AI output length: {len(enhanced_text)}")
        print(f"[DEBUG] Joined AI output (first 100 chars): '{enhanced_text[:100]}...'")
        print("[INFO] Replicate API call successful.")
        return enhanced_text.strip(), None # Return cleaned text and no error

    except replicate.exceptions.ModelError as e:
        # Specific error if the model itself failed on Replicate
        error_msg = f"Replicate Model Error: {e}"
        print(f"[ERROR] {error_msg}")
        return text, error_msg # Return original text
    except replicate.exceptions.ReplicateError as e:
        # General Replicate API errors (auth, billing, etc.)
        error_msg = f"Replicate API Error: {e}"
        print(f"[ERROR] {error_msg}")
        return text, error_msg # Return original text
    except Exception as e:
        # Catch-all for other unexpected errors (network, etc.)
        error_msg = f"Unexpected AI Enhancement Error: {str(e)}"
        print(f"[ERROR] {error_msg}", exc_info=True) # Log traceback for unexpected errors
        return text, error_msg # Return original text
    finally:
        # Clean up the environment variable after the call
        if 'REPLICATE_API_TOKEN' in os.environ:
             del os.environ['REPLICATE_API_TOKEN']
             print("[DEBUG] Cleared REPLICATE_API_TOKEN environment variable.")


# --- Main Flask Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Determine AI configuration status based on the hardcoded token
    ai_configured = bool(REPLICATE_API_TOKEN and "..." not in REPLICATE_API_TOKEN)
    print(f"[DEBUG] AI Configured Status at request start: {ai_configured}")

    # Initialize context for rendering the template
    template_context = {
        'original_filename': None,
        'processed_filename': None,
        'selected_filter': request.form.get('filter', 'adaptive_threshold') if request.method == 'POST' else 'adaptive_threshold',
        'raw_ocr_text': None,
        'ai_enhanced_text': None,
        'ocr_error': None,
        'ai_error': None,
        'ai_client_configured': ai_configured # Pass status to template
    }

    if request.method == 'POST':
        # 1. --- File Handling ---
        if 'file' not in request.files:
            flash('No file part provided.', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            original_path = None # Define for potential use in cleanup
            processed_path = None
            try:
                # Generate unique filenames to avoid collisions
                original_filename_base = str(uuid.uuid4())
                original_extension = file.filename.rsplit('.', 1)[1].lower()
                original_filename = f"{original_filename_base}.{original_extension}"
                processed_filename = f"{original_filename_base}_processed.jpg" # Standardize processed output

                original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

                # Save the uploaded file
                file.save(original_path)

                # 2. --- Image Loading ---
                image_cv = cv2.imread(original_path)
                if image_cv is None:
                    flash('Could not read uploaded image file (might be corrupted or invalid format).', 'error')
                    if os.path.exists(original_path): os.remove(original_path) # Cleanup
                    return redirect(request.url)

                # Get selected filter from form
                template_context['selected_filter'] = request.form.get('filter', 'adaptive_threshold')

                # 3. --- Image Processing (Scan/Warp/Filter) ---
                processed_image, _, img_proc_error_msg = process_document(image_cv, template_context['selected_filter'])

                if img_proc_error_msg:
                    flash(f'Image Processing Error: {img_proc_error_msg}', 'error')
                    if os.path.exists(original_path): os.remove(original_path) # Cleanup
                    # Don't redirect here, render template showing the error (optional)
                    # return redirect(request.url) # Or redirect if preferred
                    template_context['original_filename'] = original_filename # Show original even if processing failed
                    return render_template('index.html', **template_context) # Show error on page

                if processed_image is None: # Should be caught by error msg, but double-check
                     flash('Image processing failed unexpectedly.', 'error')
                     if os.path.exists(original_path): os.remove(original_path) # Cleanup
                     return redirect(request.url)

                # Save the successfully processed image
                save_success = cv2.imwrite(processed_path, processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not save_success:
                     flash('Failed to save the processed image.', 'error')
                     # Decide how to handle this - maybe continue without processed image?
                else:
                    template_context['processed_filename'] = processed_filename

                template_context['original_filename'] = original_filename

                # 4. --- Perform OCR ---
                # Use the 'processed_image' as input for OCR
                raw_ocr_text, ocr_err = perform_ocr(processed_image)
                print(f"[DEBUG] Raw OCR Text after call in route: '{raw_ocr_text[:100] if raw_ocr_text else 'None'}...'")
                print(f"[DEBUG] OCR Error after call in route: {ocr_err}")
                template_context['raw_ocr_text'] = raw_ocr_text
                template_context['ocr_error'] = ocr_err
                if ocr_err:
                    # Flash as warning, but proceed to show results including the error
                    flash(f'OCR Warning: {ocr_err}', 'warning')

                # 5. --- Enhance with AI (Llama via Replicate) ---
                # Only proceed if OCR didn't fail catastrophically (Tesseract found),
                # produced some text (even if potentially bad), and AI is configured.
                if raw_ocr_text is not None and not ocr_err and ai_configured: # Check raw_ocr_text is not None
                    if raw_ocr_text.strip(): # Check if OCR text isn't just whitespace
                        print("[INFO] Conditions met. Proceeding with AI enhancement call...")
                        ai_enhanced_text, ai_err = enhance_text_with_ai(raw_ocr_text)
                        print(f"[DEBUG] AI Enhanced Text after call in route: '{ai_enhanced_text[:100] if ai_enhanced_text else 'None'}...'")
                        print(f"[DEBUG] AI Error after call in route: {ai_err}")
                        template_context['ai_enhanced_text'] = ai_enhanced_text
                        template_context['ai_error'] = ai_err
                        if ai_err:
                            # Flash AI errors as warnings
                            flash(f'AI Enhancement Warning: {ai_err}', 'warning')
                    else:
                        template_context['ai_error'] = "AI processing skipped (OCR produced empty text)."
                        print("[INFO] AI processing skipped: OCR text was empty or whitespace.")
                # Handle cases where AI wasn't called
                elif not ai_configured:
                    template_context['ai_error'] = "AI processing skipped (Replicate API token not configured in code)."
                    print("[INFO] AI processing skipped: Replicate token missing or placeholder.")
                elif ocr_err: # If OCR had a fatal error (like Tesseract not found)
                     template_context['ai_error'] = "AI processing skipped due to OCR error."
                     print("[INFO] AI processing skipped: OCR error occurred.")
                else: # If raw_ocr_text was None (e.g., empty input image to OCR)
                    template_context['ai_error'] = "AI processing skipped (No text could be extracted by OCR)."
                    print("[INFO] AI processing skipped: OCR did not return text.")


                flash('Processing complete!', 'success')
                print(f"[DEBUG] Final Template Context before render: {template_context}")
                # Render the template with all collected results and errors
                return render_template('index.html', **template_context)

            except Exception as e:
                # Catch-all for unexpected errors during the POST request handling
                flash(f'An unexpected error occurred during processing: {str(e)}', 'error')
                print(f"[ERROR] Unexpected error in POST handler: {e}", exc_info=True) # Log full traceback
                # Attempt to clean up any files created before the error
                if original_path and os.path.exists(original_path): os.remove(original_path)
                if processed_path and os.path.exists(processed_path): os.remove(processed_path)
                return redirect(request.url) # Redirect back to the upload form on major error
        else:
            # If file extension is not allowed
            flash('Invalid file type. Allowed types: png, jpg, jpeg, webp', 'error')
            return redirect(request.url)

    # --- GET Request Handling ---
    # Just render the initial page with default context
    print("[DEBUG] Handling GET request, rendering initial page.")
    return render_template('index.html', **template_context)

# --- Routes to Serve Uploaded and Processed Images ---
@app.route('/uploads/<path:filename>') # Use path converter for safety
def serve_original_image(filename):
    print(f"[DEBUG] Serving original file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<path:filename>') # Use path converter for safety
def serve_processed_image(filename):
    print(f"[DEBUG] Serving processed file: {filename}")
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# --- Route to Download Processed Image ---
@app.route('/download/<path:filename>') # Use path converter for safety
def download_processed_image(filename):
    print(f"[DEBUG] Downloading processed file: {filename}")
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

# --- Main Execution Point ---
if __name__ == '__main__':
    # Startup check for the API token placeholder
    if not REPLICATE_API_TOKEN or "..." in REPLICATE_API_TOKEN:
        print("\n" + "*"*60)
        print("! WARNING: Replicate API token is missing or using the placeholder !")
        print("! Please edit app.py and replace 'your_r8_..._token_here' !")
        print("! AI text enhancement feature will be disabled.              !")
        print("*"*60 + "\n")

    # Run the Flask development server
    # Set debug=False for production environments
    app.run(host='0.0.0.0', port=5000, debug=True)