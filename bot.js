const mineflayer = require('mineflayer');

const bot = mineflayer.createBot({
  host: 'MQTADA12.aternos.me',
  port: 55865,
  username: 'MyBot',
  version: '1.21.1'
});

bot.on('spawn', () => {
  console.log('✅ دخلت للسيرفر!');
  bot.chat('هلو شباب 😎');
});

bot.on('chat', (username, message) => {
  if (username === bot.username) return;
  bot.chat(`سمعتك ${username} تقول: ${message}`);
});

bot.on('kicked', console.log);
bot.on('error', console.log);
