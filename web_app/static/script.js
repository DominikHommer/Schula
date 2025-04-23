document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatus = document.getElementById('upload-status');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const messagesDiv = document.getElementById('messages');
    const inputArea = document.getElementById('input-area'); // Container for input/button

    // --- Initialize Socket.IO ---
    // It will automatically connect to the server that served the page
    const socket = io();
    let isConnected = false;
    let isReadyToChat = false;

    // --- Helper Function to Add Messages ---
    function addMessage(sender, text, className = '') {
        const p = document.createElement('p');
        if (className) {
            p.classList.add(className);
        }

        // Create elements to prevent basic HTML injection
        const strong = document.createElement('strong');
        strong.textContent = sender + ': ';
        p.appendChild(strong);
        p.appendChild(document.createTextNode(text)); // Use textNode for safety

        messagesDiv.appendChild(p);
        // Scroll to the bottom
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    // --- File Upload Logic ---
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default page reload
        uploadStatus.textContent = 'Uploading and processing...';
        uploadButton.disabled = true;
        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/upload', { // Send to the Flask /upload route
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                uploadStatus.textContent = `Success: ${result.message || 'Files processed.'} Waiting for connection...`;
                addMessage('System', 'Files processed successfully.', 'system-message');
                // Don't enable chat here yet, wait for socket 'connection_ready'
            } else {
                uploadStatus.textContent = `Error: ${result.message || 'Upload failed.'}`;
                addMessage('Error', `File upload/processing failed: ${result.message || 'Unknown error'}`, 'error-message');
                uploadButton.disabled = false; // Re-enable button on failure
            }
        } catch (error) {
            console.error('Upload Fetch Error:', error);
            uploadStatus.textContent = 'Error: Could not reach server for upload.';
            addMessage('Error', 'Network error during file upload.', 'error-message');
            uploadButton.disabled = false; // Re-enable button on failure
        }
    });

    // --- Socket.IO Event Handlers ---
    socket.on('connect', () => {
        isConnected = true;
        console.log('Socket.IO connected:', socket.id);
        addMessage('System', 'Connected to server.', 'system-message');
        // If files were already processed, server might send connection_ready soon
        if (uploadStatus.textContent.startsWith('Success')) {
             uploadStatus.textContent += " Connected.";
        } else {
             uploadStatus.textContent = "Connected. Please upload files.";
        }
    });

    socket.on('disconnect', () => {
        isConnected = false;
        isReadyToChat = false;
        console.log('Socket.IO disconnected');
        addMessage('System', 'Disconnected from server. Please refresh or try again.', 'error-message');
        chatInput.disabled = true;
        sendButton.disabled = true;
        uploadStatus.textContent = "Disconnected. Refresh may be needed.";
    });

    socket.on('connection_ready', (data) => {
        console.log('Server ready:', data.message);
        addMessage('System', data.message, 'system-message');
        // NOW enable chat input
        isReadyToChat = true;
        chatInput.disabled = false;
        sendButton.disabled = false;
        uploadStatus.textContent = "Ready to chat!";
        chatInput.focus(); // Focus input field
    });

    socket.on('llm_response', (data) => {
        console.log('LLM response received:', data.answer);
        addMessage('LLM', data.answer, 'llm-response');
        // Re-enable input after getting response (optional, good UX)
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    });

    socket.on('error', (data) => {
        console.error('Server error:', data.message);
        addMessage('Error', `Server Error: ${data.message}`, 'error-message');
        // Re-enable input even if there was an error
        chatInput.disabled = false;
        sendButton.disabled = false;
    });

    // --- Sending Chat Messages ---
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message || !isConnected || !isReadyToChat) {
             console.log("Cannot send: Not connected or not ready.");
             return; // Don't send empty messages or if not ready
        }

        addMessage('You', message, 'user-message'); // Display user message immediately
        console.log('Sending message:', message);

        // Disable input while waiting for response (optional, good UX)
        chatInput.disabled = true;
        sendButton.disabled = true;

        socket.emit('chat_message', { message: message }); // Send message via WebSocket
        chatInput.value = ''; // Clear input field AFTER sending
    }

    // Event listeners for sending messages
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // --- Initial State ---
    // Clear placeholder message on load
    messagesDiv.innerHTML = '';
    addMessage('System', 'Please upload test and solution files using the form above.', 'system-message');

}); // End DOMContentLoaded