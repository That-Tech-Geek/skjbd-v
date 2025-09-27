// A simple client-side state machine
const AppState = {
    _state: {
        user: null,
        files: [],
        chunks: [],
        outline: [],
        finalNotes: [],
        currentSessionId: null,
    },
    getState() {
        return this._state;
    },
    updateState(newState) {
        this._state = { ...this._state, ...newState };
        console.log("State updated:", this._state);
    },
    resetSession() {
        this.updateState({
            files: [],
            chunks: [],
            outline: [],
            finalNotes: [],
            currentSessionId: null,
        });
        document.getElementById('file-preview-list').innerHTML = '';
        document.getElementById('outline-textarea').value = '';
        document.getElementById('synthesis-instructions').value = '';
    }
};

// --- UI Management ---
const UIManager = {
    showView(viewId) {
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active-view'));
        document.getElementById(viewId).classList.add('active-view');
    },
    showTool(toolId) {
        document.querySelectorAll('.tool-content').forEach(t => t.classList.remove('active-tool'));
        document.getElementById(toolId + '-tool').classList.add('active-tool');
    },
    showState(stateId) {
        document.querySelectorAll('.state').forEach(s => s.classList.remove('active-state'));
        document.getElementById(stateId + '-state').classList.add('active-state');
    },
    showLoader(message) {
        document.getElementById('loader-message').textContent = message;
        document.getElementById('loader').classList.remove('hidden');
    },
    hideLoader() {
        document.getElementById('loader').classList.add('hidden');
    },
    displayUserInfo(user) {
        document.getElementById('user-avatar').src = user.picture;
        document.getElementById('user-name').textContent = user.given_name;
        this.showView('main-app-view');
    },
    renderFilePreviews() {
        const fileList = document.getElementById('file-preview-list');
        fileList.innerHTML = '';
        AppState.getState().files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span>${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                <button data-index="${index}">&times;</button>
            `;
            fileList.appendChild(fileItem);
        });
        document.getElementById('process-files-button').disabled = AppState.getState().files.length === 0;
    },
    renderOutline() {
        const outline = AppState.getState().outline;
        const textarea = document.getElementById('outline-textarea');
        textarea.value = outline.map(item => item.topic).join('\n');
    },
    renderResults() {
        const notes = AppState.getState().finalNotes;
        const topicsList = document.getElementById('results-topics-list');
        topicsList.innerHTML = '';
        notes.forEach((note, index) => {
            const button = document.createElement('button');
            button.className = 'topic-button';
            button.dataset.index = index;
            button.textContent = note.topic;
            topicsList.appendChild(button);
        });
        if (notes.length > 0) {
            this.showNoteContent(0);
        }
    },
    showNoteContent(index) {
        document.querySelectorAll('.topic-button').forEach(b => b.classList.remove('active'));
        document.querySelector(`.topic-button[data-index='${index}']`).classList.add('active');
        
        const note = AppState.getState().finalNotes[index];
        // Note: For a production app, sanitize this HTML before injecting.
        document.getElementById('formatted-content').innerHTML = `<h3>${note.topic}</h3><div>${note.content}</div>`;
        document.getElementById('source-content').textContent = note.source_chunks.join('\n\n---\n\n');
    }
};

// --- API Client ---
const API = {
    async call(endpoint, method = 'GET', body = null, token) {
        const options = {
            method,
            headers: {
                'Authorization': `Bearer ${token}`
            }
        };
        if (body) {
            if (body instanceof FormData) {
                // Let the browser set the Content-Type for FormData
            } else {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(body);
            }
        }
        try {
            const response = await fetch(`/api/${endpoint}`, options);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API Error on ${endpoint}:`, error);
            alert(`An error occurred: ${error.message}`);
            throw error;
        }
    }
};

// --- Event Handlers ---
function setupEventHandlers() {
    // Auth
    document.getElementById('login-button').addEventListener('click', () => {
        window.location.href = '/api/auth-google';
    });
    document.getElementById('logout-button').addEventListener('click', () => {
        localStorage.removeItem('jwt_token');
        window.location.reload();
    });

    // Tool selection
    document.querySelectorAll('input[name="tool"]').forEach(radio => {
        radio.addEventListener('change', (e) => UIManager.showTool(e.target.value));
    });

    // File Upload
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.backgroundColor = '#e9f3ff'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.backgroundColor = 'var(--light-blue)'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = 'var(--light-blue)';
        const files = [...e.dataTransfer.files];
        const currentFiles = AppState.getState().files;
        AppState.updateState({ files: [...currentFiles, ...files] });
        UIManager.renderFilePreviews();
    });
    fileInput.addEventListener('change', () => {
        const files = [...fileInput.files];
        const currentFiles = AppState.getState().files;
        AppState.updateState({ files: [...currentFiles, ...files] });
        UIManager.renderFilePreviews();
    });
    document.getElementById('file-preview-list').addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') {
            const index = parseInt(e.target.dataset.index, 10);
            const currentFiles = AppState.getState().files;
            currentFiles.splice(index, 1);
            AppState.updateState({ files: currentFiles });
            UIManager.renderFilePreviews();
        }
    });

    // Main Engine Flow
    document.getElementById('process-files-button').addEventListener('click', handleProcessFiles);
    document.getElementById('synthesize-button').addEventListener('click', handleSynthesizeNotes);
    document.getElementById('new-session-button').addEventListener('click', () => {
        AppState.resetSession();
        UIManager.showState('upload');
    });
    document.getElementById('back-to-workspace-button').addEventListener('click', () => UIManager.showState('workspace'));

    // Results View
    document.getElementById('results-topics-list').addEventListener('click', (e) => {
        if (e.target.classList.contains('topic-button')) {
            UIManager.showNoteContent(parseInt(e.target.dataset.index, 10));
        }
    });
}

// --- Main Logic Functions ---
async function handleProcessFiles() {
    const { files } = AppState.getState();
    if (files.length === 0) return;

    UIManager.showLoader('Uploading and processing files...');
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    try {
        const token = localStorage.getItem('jwt_token');
        const result = await API.call('process-files', 'POST', formData, token);
        AppState.updateState({ chunks: result.chunks });
        
        UIManager.showLoader('Generating content outline...');
        const outlineResult = await API.call('generate-outline', 'POST', { chunks: result.chunks }, token);
        AppState.updateState({ outline: outlineResult.outline });
        
        UIManager.renderOutline();
        UIManager.showState('workspace');
    } catch (error) {
        console.error("Failed during file processing or outline generation:", error);
    } finally {
        UIManager.hideLoader();
    }
}

async function handleSynthesizeNotes() {
    const editedOutlineText = document.getElementById('outline-textarea').value;
    const instructions = document.getElementById('synthesis-instructions').value;
    const { chunks, outline } = AppState.getState();

    // Naive way to match edited topics back to original chunks. A more robust solution would be needed for complex edits.
    const editedTopics = editedOutlineText.split('\n').filter(t => t.trim() !== '');
    const synthesisPayload = editedTopics.map(topic => {
        const originalTopic = outline.find(o => o.topic === topic);
        return {
            topic,
            relevant_chunks_ids: originalTopic ? originalTopic.relevant_chunks : [], // Simple matching
            instructions
        };
    });

    UIManager.showLoader('Synthesizing notes...');
    try {
        const token = localStorage.getItem('jwt_token');
        const result = await API.call('synthesize-notes', 'POST', { topics: synthesisPayload, all_chunks: chunks }, token);
        AppState.updateState({ finalNotes: result.notes });
        UIManager.renderResults();
        UIManager.showState('results');
    } catch (error) {
        console.error("Failed to synthesize notes:", error);
    } finally {
        UIManager.hideLoader();
    }
}

// --- App Initialization ---
async function main() {
    setupEventHandlers();
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');

    if (token) {
        localStorage.setItem('jwt_token', token);
        // Clean URL
        window.history.replaceState({}, document.title, "/");
    }

    const jwtToken = localStorage.getItem('jwt_token');
    if (jwtToken) {
        try {
            UIManager.showLoader('Verifying session...');
            const { user } = await API.call('verify-token', 'GET', null, jwtToken);
            AppState.updateState({ user });
            UIManager.displayUserInfo(user);
        } catch (error) {
            localStorage.removeItem('jwt_token');
            UIManager.showView('landing-view');
        } finally {
            UIManager.hideLoader();
        }
    } else {
        UIManager.showView('landing-view');
    }
}

window.addEventListener('DOMContentLoaded', main);
