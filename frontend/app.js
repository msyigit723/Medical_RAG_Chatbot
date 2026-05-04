// ============================================
// TIBBI RAG CHATBOT - Frontend Uygulama
// FastAPI + RAG backend ile entegre
// ============================================

const API_BASE = '';  // Ayni sunucudan servis ediliyor
const API_CHAT = `${API_BASE}/api/chat`;
const API_HEALTH = `${API_BASE}/api/health`;

// State
let chatSessions = [];
let currentSessionId = null;
let isWaitingResponse = false;

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const welcomeScreen = document.getElementById('welcomeScreen');
const sidebar = document.getElementById('sidebar');
const menuBtn = document.getElementById('menuBtn');
const sidebarToggle = document.getElementById('sidebarToggle');
const newChatBtn = document.getElementById('newChatBtn');
const chatHistory = document.getElementById('chatHistory');
const themeToggle = document.getElementById('themeToggle');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// ============================================
// THEME
// ============================================
function initTheme() {
    const saved = localStorage.getItem('chatbot-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
}

themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('chatbot-theme', next);
});

// ============================================
// HEALTH CHECK
// ============================================
async function checkHealth() {
    try {
        const res = await fetch(API_HEALTH);
        const data = await res.json();
        
        if (data.status === 'ok') {
            statusDot.className = 'status-dot online';
            if (data.rag_ready) {
                statusText.textContent = `RAG Aktif (${data.total_documents} belge)`;
            } else {
                statusText.textContent = 'Fallback Mod';
            }
        }
    } catch (err) {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Baglanti Hatasi';
    }
}

// ============================================
// SIDEBAR
// ============================================
menuBtn.addEventListener('click', () => sidebar.classList.remove('collapsed'));
sidebarToggle.addEventListener('click', () => sidebar.classList.add('collapsed'));

document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 && !sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
        sidebar.classList.add('collapsed');
    }
});

// ============================================
// CHAT SESSIONS
// ============================================
function createSession() {
    const id = Date.now().toString();
    const session = { id, title: 'Yeni Sohbet', messages: [] };
    chatSessions.unshift(session);
    currentSessionId = id;
    renderHistory();
    clearChat();
    return session;
}

function getSession() {
    return chatSessions.find(s => s.id === currentSessionId);
}

function renderHistory() {
    const items = chatSessions.map(s => {
        const active = s.id === currentSessionId ? 'active' : '';
        return `<div class="history-item ${active}" data-id="${s.id}">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            <span class="history-item-text">${s.title}</span>
        </div>`;
    }).join('');
    chatHistory.innerHTML = `<div class="history-label">Gecmis Sohbetler</div>${items}`;

    chatHistory.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            currentSessionId = item.dataset.id;
            renderHistory();
            renderMessages();
        });
    });
}

function clearChat() {
    chatMessages.innerHTML = '';
    const ws = welcomeScreen.cloneNode(true);
    ws.style.display = 'flex';
    chatMessages.appendChild(ws);
    
    // Re-attach quick question handlers
    ws.querySelectorAll('.quick-q-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            messageInput.value = question;
            sendMessage(question);
        });
    });
}

// ============================================
// MESSAGES
// ============================================
function getTimeStr() {
    return new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
}

function formatText(text) {
    // Basic markdown formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function addMessage(text, type, metadata = {}) {
    const ws = chatMessages.querySelector('.welcome-screen');
    if (ws) ws.style.display = 'none';

    const session = getSession();
    if (session) {
        session.messages.push({ text, type, time: getTimeStr(), metadata });
        if (session.messages.length === 1 && type === 'user') {
            session.title = text.substring(0, 35) + (text.length > 35 ? '...' : '');
            renderHistory();
        }
    }

    renderSingleMessage(text, type, getTimeStr(), metadata);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderSingleMessage(text, type, time, metadata = {}) {
    const avatarContent = type === 'bot'
        ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'
        : 'S';

    const row = document.createElement('div');
    row.className = `message-row ${type}`;

    let extraHTML = '';
    
    // Show source badges for bot messages
    if (type === 'bot' && metadata.sources && metadata.sources.length > 0) {
        const badges = metadata.sources.map(s => `<span class="source-badge">${s}</span>`).join('');
        extraHTML += `<div class="source-badges">${badges}</div>`;
    }

    // Show RAG indicator
    if (type === 'bot' && metadata.source) {
        const isRag = metadata.source === 'rag';
        const label = isRag ? 'RAG ile yanitlandi' : 'Genel bilgi';
        const cls = isRag ? 'rag-indicator' : 'rag-indicator fallback';
        extraHTML += `<div class="${cls}">${label}</div>`;
    }

    row.innerHTML = `
        <div class="message-avatar">${avatarContent}</div>
        <div class="message-content">
            <div class="message-bubble">${formatText(text)}</div>
            ${extraHTML}
            <span class="message-time">${time}</span>
        </div>`;
    chatMessages.appendChild(row);
}

function showTyping() {
    const row = document.createElement('div');
    row.className = 'message-row bot';
    row.id = 'typingRow';
    row.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator"><span></span><span></span><span></span></div>
            </div>
        </div>`;
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById('typingRow');
    if (el) el.remove();
}

// ============================================
// API CALL
// ============================================
async function fetchBotResponse(question) {
    try {
        const res = await fetch(API_CHAT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: question }),
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        return {
            answer: data.answer,
            source: data.source,
            confidence: data.confidence,
            sources: data.sources || [],
        };
    } catch (err) {
        console.error('API hatasi:', err);
        return {
            answer: 'Sunucuya baglanirken bir hata olustu. Lutfen tekrar deneyin.',
            source: 'error',
            confidence: 0,
            sources: [],
        };
    }
}

// ============================================
// SEND MESSAGE
// ============================================
async function sendMessage(text) {
    if (!text.trim() || isWaitingResponse) return;
    if (!currentSessionId) createSession();

    isWaitingResponse = true;
    addMessage(text.trim(), 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;

    showTyping();

    const response = await fetchBotResponse(text.trim());

    removeTyping();
    addMessage(response.answer, 'bot', {
        source: response.source,
        sources: response.sources,
    });

    isWaitingResponse = false;
}

// ============================================
// EVENT LISTENERS
// ============================================
sendBtn.addEventListener('click', () => sendMessage(messageInput.value));

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(messageInput.value);
    }
});

messageInput.addEventListener('input', () => {
    sendBtn.disabled = !messageInput.value.trim();
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
});

newChatBtn.addEventListener('click', () => {
    createSession();
    if (window.innerWidth <= 768) sidebar.classList.add('collapsed');
});

// Quick question buttons (initial)
document.querySelectorAll('.quick-q-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const question = btn.getAttribute('data-question');
        messageInput.value = question;
        sendMessage(question);
    });
});

function renderMessages() {
    const session = getSession();
    chatMessages.innerHTML = '';
    if (!session || session.messages.length === 0) {
        clearChat();
        return;
    }
    session.messages.forEach(m => {
        renderSingleMessage(m.text, m.type, m.time, m.metadata || {});
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============================================
// INIT
// ============================================
initTheme();
if (window.innerWidth <= 768) sidebar.classList.add('collapsed');
checkHealth();

// Periyodik saglik kontrolu
setInterval(checkHealth, 30000);
