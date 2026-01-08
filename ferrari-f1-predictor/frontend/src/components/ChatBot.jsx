import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Bot, User } from 'lucide-react';

const ChatBot = ({ apiUrl }) => {
    const [messages, setMessages] = useState([
        { type: 'bot', text: 'Ciao! I am your Ferrari F1 Strategy AI. Ask me about historical performance or 2026 predictions.' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = input;
        setInput('');
        setMessages(prev => [...prev, { type: 'user', text: userMessage }]);
        setLoading(true);

        try {
            const response = await axios.post(`${apiUrl}/chat`, { query: userMessage });
            setMessages(prev => [...prev, { type: 'bot', text: response.data.response }]);
        } catch (error) {
            setMessages(prev => [...prev, { type: 'bot', text: 'Sorry, my telemetry is down. Please try again.' }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card chat-container">
            <h2><Bot className="text-ferrari-red" /> Team Radio</h2>
            <div className="messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`}>
                        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.2rem' }}>
                            {msg.type === 'bot' ? <Bot size={16} /> : <User size={16} />}
                            <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                                {msg.type === 'bot' ? 'Strategist' : 'You'}
                            </span>
                        </div>
                        {msg.text}
                    </div>
                ))}
                {loading && <div className="message bot">Thinking... 🏎️</div>}
                <div ref={messagesEndRef} />
            </div>
            <form className="chat-input" onSubmit={handleSend}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about strategy, history, or predictions..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading}>
                    <Send size={18} />
                </button>
            </form>
        </div>
    );
};

export default ChatBot;
