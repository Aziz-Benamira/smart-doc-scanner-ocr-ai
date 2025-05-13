# Smart Document Scanner & AI Text Enhancer

A Flask-based web application that provides a streamlined workflow for processing document images. Users can upload an image, which is then automatically perspective-corrected (like a scanner app). The application extracts text from the processed image using Tesseract OCR and subsequently enhances this text using a Llama language model via the Replicate API, making it cleaner and more suitable for copy-pasting.

![Screenshot (Optional - Add a screenshot of your app here if you have one)](placeholder_screenshot.png)

## Features

*   **Image Upload:** Supports common image formats (PNG, JPG, JPEG, WEBP) via drag-and-drop or file browser.
*   **Perspective Correction:** Automatically detects document boundaries and performs a 4-point perspective transform to get a flat, "scanned" view.
*   **Image Filtering:** Offers options for the processed image output (e.g., adaptive threshold for B&W, grayscale, color).
*   **OCR (Optical Character Recognition):** Extracts text from the processed document image using Tesseract OCR.
*   **AI Text Enhancement:** Sends the raw OCR text to a Llama model (via Replicate API) to:
    *   Correct common OCR errors.
    *   Improve sentence structure and flow.
    *   Format text logically with paragraphs.
*   **Side-by-Side Display:** Shows the original uploaded image, the perspective-corrected image, the raw OCR text, and the AI-enhanced text.
*   **Download & Copy:** Allows downloading the processed image and copying both raw and AI-enhanced text.
*   **User-Friendly Interface:** Built with Flask and styled with Bootstrap for a responsive experience.

## Technologies Used

*   **Backend:**
    *   Python 3
    *   Flask (Web framework)
    *   OpenCV (Image processing, perspective correction)
    *   Pytesseract (Tesseract OCR wrapper)
    *   Replicate (Python client for Llama API)
    *   NumPy (Numerical operations)
    *   Imutils (Image processing utilities)
*   **Frontend:**
    *   HTML5
    *   CSS3 (with Bootstrap 5)
    *   JavaScript (for interactivity like drag-and-drop, image preview, copy-to-clipboard)
*   **External Services/Tools:**
    *   Tesseract OCR Engine (must be installed locally)
    *   Replicate API (for Llama model access)

## Setup and Installation

### Prerequisites

1.  **Python 3.8+** installed.
2.  **Tesseract OCR Engine installed** and accessible in your system's PATH.
    *   **Windows:** Download from [UB Mannheim Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki) and ensure "Add Tesseract to system PATH" is checked during installation.
    *   **macOS:** `brew install tesseract`
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install tesseract-ocr`
    *   Verify by typing `tesseract --version` in your terminal.
3.  **Replicate API Token:** You need an API token from [Replicate](https://replicate.com/).
4.  **(Optional but Recommended) Git** for cloning the repository.

### Installation Steps

1.  **Clone the Repository (Optional):**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```
    *(If you're not using Git, ensure you have all the project files in a single directory.)*

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Token:**
    *   Open the `app.py` file.
    *   Find the line: `REPLICATE_API_TOKEN = "your_r8_..._token_here"`
    *   Replace `"your_r8_..._token_here"` with your actual Replicate API token.
    *   **(Optional) Configure Tesseract Path:** If `pytesseract` cannot find your Tesseract installation (even if `tesseract --version` works in the terminal), uncomment and set the correct path in `app.py`:
        ```python
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
        ```

5.  **Configure Flask Secret Key:**
    *   Open `app.py`.
    *   Find the line: `app.config['SECRET_KEY'] = 'your-very-own-local-secret-key-that-is-random'`
    *   Replace the placeholder string with your own unique, random string. This is important for session security if you were to deploy it.

6.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application will typically be available at `http://127.0.0.1:5000/` or `http://localhost:5000/` in your web browser.

## Usage

1.  Open the application in your web browser.
2.  Drag and drop an image file onto the designated area, or click "Browse Files" to select an image.
3.  (Optional) Select an output filter for the processed image.
4.  Click "Scan Document".
5.  The application will process the image, display the original and processed versions, and show the raw OCR text alongside the AI-enhanced text.
6.  Use the "Download" button to save the processed image and the "Copy" buttons to copy the extracted texts.

## Project Structure
smart-doc-scanner-ocr-ai/
├── app.py # Main Flask application logic

├── requirements.txt # Python dependencies

├── templates/

│ └── index.html # HTML template for the UI

├── static/

│ ├── css/

│ │ └── style.css # Custom CSS styles

│ └── js/

│ └── script.js # JavaScript for frontend interactivity

├── uploads/ # (Created automatically) Directory for uploaded images

├── processed/ # (Created automatically) Directory for processed images

└── README.md # This file
## Choosing a Llama Model on Replicate

The `app.py` file specifies a Llama model identifier to use with the Replicate API.

# Inside enhance_text_with_ai function in app.py
model_identifier = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d" # Example
Use code with caution.
You can change this to use other Llama models available on Replicate (e.g., Llama 3 versions).
Go to Replicate Explore.
Find the Llama model you wish to use.
Click on the model, then go to its "API" tab.
Copy the model identifier string (e.g., meta/meta-llama-3-8b-instruct:xxxxxxxx...).
Replace the model_identifier value in app.py with the new one.
You may also need to adjust the prompt_text format within the enhance_text_with_ai function to match the optimal prompting style for the chosen Llama model version.
Future Enhancements / To-Do
Allow users to select different Llama models from the UI.
Implement asynchronous processing for long OCR/AI tasks to prevent UI blocking.
Add more advanced image pre-processing options before OCR.
Improve error handling and user feedback.
Support for PDF uploads.
User accounts and history of processed documents.
Contributing
Contributions are welcome! If you have suggestions for improvements or find any bugs, please feel free to open an issue or submit a pull request.


