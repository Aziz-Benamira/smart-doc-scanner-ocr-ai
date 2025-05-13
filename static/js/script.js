document.addEventListener('DOMContentLoaded', function() {

    // --- Get DOM Elements ---
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file');
    const uploadArea = document.getElementById('upload-area');
    const browseButton = document.getElementById('browse-button');
    const filenameDisplay = document.getElementById('filename-display');
    const submitButton = document.getElementById('submit-button');
    const loadingSpinner = submitButton?.querySelector('.spinner-border'); // Use optional chaining
    const submitButtonIcon = submitButton?.querySelector('.submit-icon');
    const submitButtonText = submitButton?.querySelector('.submit-text');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');

    // --- Event Listeners ---

    // Trigger hidden file input when "Browse" button is clicked
    if (browseButton) {
        browseButton.addEventListener('click', () => {
            fileInput?.click(); // Use optional chaining
        });
    }

    // Trigger hidden file input when the main upload area is clicked (but not the button inside it)
    if (uploadArea) {
        uploadArea.addEventListener('click', (e) => {
            if (browseButton && e.target !== browseButton && !browseButton.contains(e.target)) {
               fileInput?.click();
            }
        });
    }

    // Handle file selection (from browse or drag/drop)
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // --- Drag and Drop Handling ---
    if (uploadArea) {
        // Prevent default browser behavior for drag/drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false); // Also prevent for body
        });

        // Highlight upload area when item is dragged over
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });

        // Remove highlight when item leaves or is dropped
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });

        // Handle file drop
        uploadArea.addEventListener('drop', (event) => {
            if (event.dataTransfer && event.dataTransfer.files) {
                fileInput.files = event.dataTransfer.files; // Assign dropped files to hidden input
                handleFileSelect({ target: fileInput }); // Process the dropped file(s) using the common handler
            }
        });
    }

    // Handle form submission: show loading state
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            // Basic validation: Ensure a file is selected
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                 flashMessage("Please select an image file before scanning.", "danger"); // Use custom flash
                 highlightElement(uploadArea, 'border-danger'); // Highlight upload area
                 event.preventDefault(); // Stop form submission
                 return;
            }
            resetHighlight(uploadArea, 'border-danger'); // Reset highlight if validation passes

            // Show loading indicator on submit button
            if (loadingSpinner) loadingSpinner.style.display = 'inline-block';
            if (submitButtonIcon) submitButtonIcon.style.display = 'none'; // Hide icon
            if (submitButtonText) submitButtonText.textContent = 'Processing...'; // Change text
            if (submitButton) submitButton.disabled = true; // Disable button
        });
    }

    // --- Clipboard Copy Functionality ---
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetSelector = button.getAttribute('data-clipboard-target');
            const targetElement = document.querySelector(targetSelector);

            if (targetElement) {
                const textToCopy = targetElement.textContent || targetElement.innerText; // Get text

                navigator.clipboard.writeText(textToCopy).then(() => {
                    // Success feedback: Change button appearance temporarily
                    const originalContent = button.innerHTML;
                    button.innerHTML = '<i class="fas fa-check me-1"></i> Copied!';
                    button.classList.add('btn-success', 'text-white'); // Use text-white for contrast
                    button.classList.remove('btn-outline-secondary');

                    // Revert after a delay
                    setTimeout(() => {
                        button.innerHTML = originalContent;
                        button.classList.remove('btn-success', 'text-white');
                        button.classList.add('btn-outline-secondary');
                    }, 2000); // 2 seconds

                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                    // Error feedback (consider a less intrusive way than alert)
                    flashMessage("Failed to copy text to clipboard.", "warning");
                });
            } else {
                console.error('Clipboard target not found:', targetSelector);
            }
        });
    });

    // --- Helper Functions ---

    // Common file processing logic
    function handleFileSelect(event) {
        const files = event.target.files; // Directly use event.target.files
        resetHighlight(uploadArea, 'border-danger'); // Reset validation highlight

        if (files && files.length > 0) {
            const file = files[0];
            // Basic type validation
            if (file.type.startsWith('image/')) {
                filenameDisplay.textContent = `Selected: ${file.name}`;
                showImagePreview(file);
            } else {
                 filenameDisplay.textContent = 'Invalid file type. Please select an image.';
                 hideImagePreview();
                 fileInput.value = ''; // Clear invalid selection from input
                 highlightElement(uploadArea, 'border-danger');
            }
        } else {
            // Handle cases where file selection is cancelled or empty
            hideImagePreview(); // Also clears filename display
        }
    }

    // Display image preview
    function showImagePreview(file) {
         if (!imagePreview || !imagePreviewContainer) return;
         const reader = new FileReader();
         reader.onload = function(e) {
             imagePreview.src = e.target.result;
             imagePreviewContainer.style.display = 'block'; // Show container
         }
         reader.readAsDataURL(file);
    }

    // Hide image preview
     function hideImagePreview() {
         if (imagePreviewContainer) imagePreviewContainer.style.display = 'none'; // Hide container
         if (imagePreview) imagePreview.src = '#'; // Reset src
         if (filenameDisplay) filenameDisplay.textContent = ''; // Clear filename
         // Note: fileInput.value is cleared in handleFileSelect if needed
     }

    // Prevent default drag/drop behavior
    function preventDefaults (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    // Simple function to add/remove highlight class
    function highlightElement(element, className) {
        if (element) element.classList.add(className);
    }
    function resetHighlight(element, className) {
        if (element) element.classList.remove(className);
    }

     // Simple Flash Message Function (using Bootstrap dynamically)
     function flashMessage(message, type = 'info') {
         // Try to find a good place to insert the alert, e.g., after the header
         const container = document.querySelector('.container');
         const header = container?.querySelector('header');
         const targetElement = header || container?.firstChild; // Insert after header or at top

         if (!targetElement) return; // Can't find where to put the message

         const alertDiv = document.createElement('div');
         const bootstrapClass = (type === 'error' || type === 'danger') ? 'danger' : (type === 'warning' ? 'warning' : 'success');
         const iconClass = (type === 'error' || type === 'danger') ? 'fa-exclamation-triangle' : (type === 'warning' ? 'fa-exclamation-circle' : 'fa-check-circle');

         alertDiv.className = `alert alert-${bootstrapClass} alert-dismissible fade show shadow-sm mt-3 dynamic-alert`;
         alertDiv.setAttribute('role', 'alert');
         alertDiv.innerHTML = `
             <i class="fas ${iconClass} me-2"></i>
             ${message}
             <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
         `;

         // Insert the alert into the DOM
         targetElement.parentNode.insertBefore(alertDiv, targetElement.nextSibling);

         // Auto-dismiss after 5 seconds (optional)
         setTimeout(() => {
             const alertInstance = bootstrap.Alert.getOrCreateInstance(alertDiv);
             if(alertInstance) {
                 alertInstance.close();
             } else {
                 // Fallback if Bootstrap JS isn't loaded or fails
                 alertDiv.remove();
             }
         }, 5000);
     }

}); // End DOMContentLoaded