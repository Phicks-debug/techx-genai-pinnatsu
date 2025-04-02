document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const connectionStatus = document.getElementById('connection-status');

    let socket = null;
    let userId = generateUserId();
    let currentAiMessage = null; // To track the current AI message element being streamed
    let currentAiMarkdown = ''; // Accumulate raw Markdown for the current streaming message

    console.log("Chat application starting with userId:", userId);

    // Configure DOMPurify
    DOMPurify.setConfig({ USE_PROFILES: { html: true } });

    // Connect to WebSocket
    function connectWebSocket() {
        updateConnectionStatus('connecting');
        console.log("Attempting to connect to WebSocket server...");
        socket = new WebSocket('ws://localhost:8000/ws');

        socket.onopen = () => {
            console.log("WebSocket connection established!");
            updateConnectionStatus('connected');
            addSystemMessage('Connected to chat server');
        };

        socket.onmessage = (event) => {
            console.log("Raw message received:", event.data);

            try {
                const data = JSON.parse(event.data);
                console.log("Parsed message:", data);

                // --- Handle streaming format ---
                if (data.chunk && data.chunk.delta && data.chunk.delta.type === 'text_delta') {
                    const textChunk = data.chunk.delta.text;
                    console.log("Received text chunk:", textChunk);

                    // Append to current AI message or create new one
                    if (!currentAiMessage) {
                        // Reset accumulated Markdown for the new message
                        currentAiMarkdown = '';
                        // Create new message element
                        currentAiMessage = document.createElement('div');
                        currentAiMessage.className = 'message ai-message';
                        chatBox.appendChild(currentAiMessage);
                    }

                    // Append new text chunk to accumulated Markdown
                    currentAiMarkdown += textChunk;

                    // Parse the *entire accumulated* Markdown and sanitize
                    const unsafeHtml = marked.parse(currentAiMarkdown);
                    currentAiMessage.innerHTML = DOMPurify.sanitize(unsafeHtml);

                    scrollToBottom();
                }
                // --- Handle old format (non-streaming) for backward compatibility ---
                else if (data.message) {
                    // Check if this is our own message echo (shouldn't happen with current server logic, but safe)
                    if (data.user_id === userId) {
                        console.log("Skipping own message echo (old format)");
                        return;
                    }

                    console.log("Received complete message (old format):", data.message);

                    // Reset streaming state
                    currentAiMessage = null;
                    currentAiMarkdown = '';

                    // Display the complete message using Markdown
                    addMessage(data.message, 'ai');
                }
                // --- Handle end of stream or unknown format ---
                else {
                    console.log("Unknown message format or potentially end of stream signal");
                    // Finalize the current message (if any) and reset for the next one
                    currentAiMessage = null;
                    currentAiMarkdown = '';
                }
            } catch (error) {
                console.error('Error parsing message:', error);
                addSystemMessage('Error processing message from server');
                addSystemMessage('Raw data: ' + event.data);
                // Reset streaming state on error
                currentAiMessage = null;
                currentAiMarkdown = '';
            }
        };

        socket.onclose = (event) => {
            console.log("WebSocket closed with code:", event.code, "reason:", event.reason);
            updateConnectionStatus('disconnected');
            addSystemMessage('Disconnected from chat server');
            // Reset streaming state on disconnect
            currentAiMessage = null;
            currentAiMarkdown = '';

            setTimeout(() => {
                if (!socket || socket.readyState === WebSocket.CLOSED) {
                    console.log("Attempting to reconnect...");
                    connectWebSocket();
                }
            }, 5000);
        };

        socket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            addSystemMessage('Error with connection. Trying to reconnect...');
            // Reset streaming state on error
            currentAiMessage = null;
            currentAiMarkdown = '';
        };
    }

    // Send message to server
    function sendMessage() {
        const message = messageInput.value.trim();

        // Reset the current AI message state as we're sending a new query
        currentAiMessage = null;
        currentAiMarkdown = '';

        if (message && socket && socket.readyState === WebSocket.OPEN) {
            const messageObj = {
                user_id: userId,
                messages: [{
                    role: "user",
                    content: message,
                }],
                timestamp: new Date().toISOString()
            };

            try {
                const jsonStr = JSON.stringify(messageObj);
                console.log("Sending message:", jsonStr);
                socket.send(jsonStr);

                // Display sent message (user messages don't need Markdown)
                addMessage(message, 'user');

                messageInput.value = '';
            } catch (error) {
                console.error("Error sending message:", error);
                addSystemMessage('Failed to send message: ' + error.message);
            }
        } else if (!socket || socket.readyState !== WebSocket.OPEN) {
            console.warn("Cannot send message - WebSocket not connected");
            addSystemMessage('Not connected to server. Trying to reconnect...');
            if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
                connectWebSocket();
            }
        }
    }

    // Add message to chat box
    function addMessage(text, sender) {
        console.log(`Adding ${sender} message:`, text);
        const messageElement = document.createElement('div'); // Use div for consistency
        messageElement.className = `message ${sender}-message`;

        if (sender === 'ai') {
            // Parse AI messages as Markdown and sanitize
            const unsafeHtml = marked.parse(text);
            messageElement.innerHTML = DOMPurify.sanitize(unsafeHtml);
        } else {
            // User messages are plain text
            messageElement.textContent = text;
        }

        chatBox.appendChild(messageElement);
        scrollToBottom();
    }

    // Add system message to chat box
    function addSystemMessage(text) {
        console.log("System message:", text);
        const messageElement = document.createElement('div');
        messageElement.className = 'system-message';
        messageElement.textContent = text; // System messages are plain text

        chatBox.appendChild(messageElement);
        scrollToBottom();
    }

    // Update connection status indicator
    function updateConnectionStatus(status) {
        console.log("Connection status updated to:", status);
        connectionStatus.className = `connection-status ${status}`;
        connectionStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    // Scroll chat box to bottom
    function scrollToBottom() {
        // Add a small delay to allow the DOM to update, especially with images/complex markdown
        setTimeout(() => {
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 50);
    }

    // Generate a random user ID
    function generateUserId() {
        return 'user_' + Math.random().toString(36).substring(2, 10);
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial connection
    connectWebSocket();
});
