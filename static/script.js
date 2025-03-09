document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("upload-form");
    const loadingMessage = document.getElementById("loading");

    form.addEventListener("submit", function() {
        loadingMessage.classList.remove("hidden");
    });
});
