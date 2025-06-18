const mineflayer = require('mineflayer');
const util = require('minecraft-server-util');

const serverAddress = 'MQTADA12.aternos.me';
const serverPort = 55865;

const maxBots = 20;
const botLifetime = 100 * 60 * 1000; // 100 دقيقة
const baseInterval = 300; // 5 دقائق
const maxInterval = 360; // 6 دقائق

let serverVersion = 'auto';       // سيُحَدَّد تلقائيًا لاحقًا
let activeBots = new Array(maxBots).fill(null);
let scheduleDelays = new Array(maxBots).fill(0);

// استعلام عن إصدار السرفر وتخزينه
async function detectServerVersion() {
  try {
    const status = await util.status(serverAddress, { port: serverPort });
    console.log(`🛰️ تم اكتشاف إصدار السرفر: ${status.version.name}`);
    serverVersion = status.version.name;
  } catch (err) {
    console.warn('⚠️ فشل اكتشاف إصدار السرفر، سيتم استخدام الإصدار الأحدث تلقائيًا.');
    serverVersion = 'auto';
  }
}

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
  for (let i = 0; i < 8; i++) {  // أطول قليلًا لضمان التفرد
    name += chars[Math.floor(Math.random() * chars.length)];
  }
  return name;
}

function createBot(botIndex) {
  const botName = generateRandomName();
  const bot = mineflayer.createBot({
    host: serverAddress,
    port: serverPort,
    username: botName,
    version: serverVersion
  });

  bot.on('spawn', () => {
    console.log(`✅ ${bot.username} دخل السيرفر بإصدار ${serverVersion}.`);
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

    // في حال أخطاء الاتصال، نؤجل جميع الجداول دقيقة واحدة
    if (
      err.code === 'ECONNREFUSED' ||
      err.code === 'ECONNRESET' ||
      (err.message && err.message.includes('read ECONNRESET'))
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

// البداية: اكتشاف الإصدار ثم جدولة البوتات
(async () => {
  await detectServerVersion();
  for (let i = 0; i < maxBots; i++) {
    scheduleNextBot(i);
  }
})();
