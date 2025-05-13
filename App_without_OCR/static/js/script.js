document.addEventListener('DOMContentLoaded', function() {

    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file');
    const uploadArea = document.getElementById('upload-area');
    const browseButton = document.getElementById('browse-button');
    const filenameDisplay = document.getElementById('filename-display');
    const submitButton = document.getElementById('submit-button');
    const loadingSpinner = submitButton.querySelector('.spinner-border');
    const submitButtonIcon = submitButton.querySelector('i'); // Get the icon
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');

    // --- Browse Button ---
    browseButton.addEventListener('click', () => {
        fileInput.click(); // Trigger hidden file input click
    });

    // --- Trigger file input when upload area is clicked ---
    uploadArea.addEventListener('click', (e) => {
         // Prevent triggering if the browse button itself was clicked
        if (e.target !== browseButton && !browseButton.contains(e.target)) {
           fileInput.click();
        }
    });

    // --- Handle File Selection (via Browse or Drop) ---
    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect(event) {
        const files = event.target.files || event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
             // Basic validation (optional: add size check)
            if (file.type.startsWith('image/')) {
                filenameDisplay.textContent = `Selected: ${file.name}`;
                showImagePreview(file);
            } else {
                 filenameDisplay.textContent = 'Please select an image file.';
                 hideImagePreview();
                 fileInput.value = ''; // Clear invalid selection
            }
        }
    }

    function showImagePreview(file) {
         const reader = new FileReader();
         reader.onload = function(e) {
             imagePreview.src = e.target.result;
             imagePreviewContainer.style.display = 'block';
         }
         reader.readAsDataURL(file);
    }

     function hideImagePreview() {
         imagePreviewContainer.style.display = 'none';
         imagePreview.src = '#';
     }

    // --- Drag and Drop ---
    uploadArea.addEventListener('dragover', (event) => {
        event.preventDefault(); // Prevent default behavior (opening file)
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (event) => {
        event.preventDefault(); // Prevent default behavior
        uploadArea.classList.remove('dragover');
        fileInput.files = event.dataTransfer.files; // Assign dropped files to input
        handleFileSelect(event); // Process the dropped file
    });

    // --- Form Submission - Loading State ---
    uploadForm.addEventListener('submit', function() {
        // Ensure a file is selected
        if (!fileInput.files || fileInput.files.length === 0) {
             alert("Please select an image file before scanning."); // Or use a flash message approach
             event.preventDefault(); // Stop form submission
             return;
        }

        // Show loading state
        if(loadingSpinner) loadingSpinner.style.display = 'inline-block';
        if(submitButtonIcon) submitButtonIcon.style.display = 'none'; // Hide original icon
        submitButton.disabled = true;
        submitButton.querySelector('span:not(.spinner-border)')?.remove(); // Remove existing text node if needed
        submitButton.insertAdjacentText('beforeend', ' Processing...'); // Add text after spinner
    });

    // --- Resetting form state if needed ---
    // If you navigate back or an error occurs without full page reload (e.g., with AJAX later),
    // you might need logic to reset the button state.
    // For now, the page reload handles the reset.

});