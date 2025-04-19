document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const messagesDiv = document.getElementById('messages');
    // const resetForm = document.getElementById('reset-form'); // Already handled by form POST

    // --- Handle File Upload ---
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        uploadStatus.textContent = 'Uploading...';
        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                uploadStatus.textContent = `Success: ${result.message || 'Files uploaded.'}`;
                // Optionally clear the form fields
                // uploadForm.reset(); 
                // Maybe refresh part of the page or enable chat here if it wasn't before
            } else {
                uploadStatus.textContent = `Error: ${result.error || 'Upload failed.'}`;
            }
        } catch (error) {
            console.error('Upload Error:', error);
            uploadStatus.textContent = 'Error: Could not reach server.';
        }
    });

    // --- Handle Chat ---
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Display user message immediately
        addMessage('You', message);
        chatInput.value = ''; // Clear input

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            const result = await response.json();

            if (response.ok) {
                addMessage('LLM', result.answer);
            } else {
                addMessage('Error', result.error || 'Failed to get response.');
            }
        } catch (error) {
            console.error('Chat Error:', error);
            addMessage('Error', 'Could not reach server.');
        }
    }

    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function addMessage(sender, text) {
        const p = document.createElement('p');
        p.innerHTML = `<strong>${sender}:</strong> ${text}`; // Use innerHTML cautiously if text could contain HTML
        messagesDiv.appendChild(p);
        messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to bottom
    }
    
    // Initial scroll to bottom if messages loaded from server
    messagesDiv.scrollTop = messagesDiv.scrollHeight; 

});