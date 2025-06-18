const mineflayer = require('mineflayer');

const serverAddress = 'MQTADA12.aternos.me';
const serverPort = 55865;

const maxBots = 20;
const botLifetime = 100 * 60 * 1000; // 100 دقيقة
const baseInterval = 300; // 5 دقائق
const maxInterval = 360; // 6 دقائق

let activeBots = new Array(maxBots).fill(null);
let scheduleDelays = new Array(maxBots).fill(0);

function getIntervalForBot(botNumber) {
  let interval = baseInterval + (botNumber % (maxInterval - baseInterval + 1));
  interval += scheduleDelays[botNumber];
  return interval * 1000;
}

function delayAllSchedulesOneMinute() {
  for (let i = 0; i < maxBots; i++) {
    scheduleDelays[i] += 1;
  }
  console.log("⏳ تأخير الجداول دقيقة بسبب فشل الاتصال.");
}

function generateRandomName() {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let name = 'Bot_';
  for (let i = 0; i < 5; i++) {
    name += chars[Math.floor(Math.random() * chars.length)];
  }
  return name;
}

function createBot(botIndex) {
  const bot = mineflayer.createBot({
    host: serverAddress,
    port: serverPort,
    username: generateRandomName(),
    version: 'auto' // ← استخدام الإصدار الأحدث تلقائيًا
  });

  bot.on('spawn', () => {
    console.log(`✅ ${bot.username} دخل السيرفر.`);
    activeBots[botIndex] = bot;

    setTimeout(() => {
      bot.quit();
      console.log(`⏱️ ${bot.username} خرج بعد 100 دقيقة.`);
      activeBots[botIndex] = null;
      scheduleNextBot(botIndex);
    }, botLifetime);
  });

  bot.on('error', (err) => {
    console.error(`❌ خطأ في ${bot.username}:`, err.code || err.message);

    if (
      err.code === 'ECONNREFUSED' ||
      err.code === 'ECONNRESET' ||
      err.message.includes('read ECONNRESET')
    ) {
      console.log(`🔁 إعادة محاولة دخول ${bot.username} بعد دقيقة.`);
      delayAllSchedulesOneMinute();
      setTimeout(() => scheduleNextBot(botIndex), 60 * 1000);
    }
  });

  bot.on('end', () => {
    console.log(`🛑 ${bot.username} انتهت جلسته.`);
    activeBots[botIndex] = null;
  });
}

function scheduleNextBot(botIndex) {
  const interval = getIntervalForBot(botIndex);
  console.log(`📆 جدولة بوت ${botIndex} للدخول بعد ${Math.floor(interval / 1000)} ثانية.`);
  setTimeout(() => {
    if (!activeBots[botIndex]) {
      createBot(botIndex);
    }
  }, interval);
}

for (let i = 0; i < maxBots; i++) {
  scheduleNextBot(i);
}
