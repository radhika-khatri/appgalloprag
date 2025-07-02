// whatsapp_bot.js

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');

const client = new Client({
  authStrategy: new LocalAuth()
});

// In-memory map to track chat history per user
const userHistory = new Map();

client.on('qr', qr => {
  qrcode.generate(qr, { small: true });
  console.log('📱 Scan this QR code to log in.');
});

client.on('ready', () => {
  console.log('✅ WhatsApp bot is ready!');
});

client.on('message', async message => {
  if (message.type !== 'chat') return;

  const question = message.body.trim();
  const userId = message.from;
  console.log(`📨 Message from ${userId}: ${question}`);

  // Handle follow-up question based on history
  if (question.toLowerCase().includes("explain the response of all the above user ids")) {
    const history = userHistory.get(userId) || [];

    if (history.length === 0) {
      await message.reply("📭 No previous history found.");
      return;
    }

    const historyText = history
      .map((e, i) => `Q${i + 1}: ${e.question}\nA${i + 1}: ${e.answer}`)
      .join("\n---\n");

    try {
      const res = await axios.post('http://localhost:8000/ask', {
        question: `Explain the following previous interactions:\n${historyText}`
      });
      const reply = res.data;
      await message.reply(`🧠 ${reply}`);
    } catch (err) {
      console.error('❌ FastAPI error (history follow-up):', err.response?.data || err.message);
      await message.reply("⚠️ Could not process history-based question.");
    }

    return;
  }

  // Ask the chatbot as usual
  try {
    const res = await axios.post('http://localhost:8000/ask', {
      question: question
    });

    const reply = res.data;

    // Update user history
    if (!userHistory.has(userId)) {
      userHistory.set(userId, []);
    }
    userHistory.get(userId).push({ question, answer: reply });

    await message.reply(`💡 ${reply}`);
  } catch (err) {
    console.error('❌ FastAPI error:', err.response?.data || err.message);
    await message.reply("⚠️ Could not answer your question. Try again later.");
  }
});

client.initialize();
