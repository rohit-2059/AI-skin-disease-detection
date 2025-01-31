// Check if file is selected before submitting
document.getElementById("uploadForm").onsubmit = function(event) {
    var fileInput = document.getElementById("file");
    if (!fileInput.files[0]) {
        alert("Please select an image to upload.");
        event.preventDefault();
    }
};
