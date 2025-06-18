const mineflayer = require('mineflayer');

const serverAddress = 'MQTADA12.aternos.me';
const serverPort = 55865;

const maxBots = 20;
const botLifetime = 100 * 60 * 1000; // 100 دقيقة (100 x 60 x 1000 ملّي)
const baseInterval = 300; // 5 دقائق بالثواني
const maxInterval = 360; // 6 دقائق بالثواني

// ثبّتنا الإصدار يدويًّا ليتوافق مع سيرفرك Java Vanilla 1.20
const serverVersion = '1.20';

let activeBots = new Array(maxBots).fill(null);
let scheduleDelays = new Array(maxBots).fill(0);

function getIntervalForBot(botNumber) {
  // نأخذ الفارق ضمن النطاق base–max ثم نضيف التأخيرات المتراكمة
  let intervalSec = baseInterval + (botNumber % (maxInterval - baseInterval + 1));
  intervalSec += scheduleDelays[botNumber];
  return intervalSec * 1000; // نحول للميلي ثانية
}

function delayAllSchedulesOneMinute() {
  for (let i = 0; i < maxBots; i++) {
    scheduleDelays[i] += 60; // أضف 60 ثانية (دقيقة) لكل جدول
  }
  console.log("⏳ تم تأخير جميع الجداول دقيقة بسبب فشل الاتصال.");
}

function generateRandomName() {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let name = 'Bot_';
  for (let i = 0; i < 8; i++) {
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
      console.log(`⏱️ ${bot.username} خرج بعد ${botLifetime / 60000} دقيقة.`);
      activeBots[botIndex] = null;
      scheduleNextBot(botIndex);
    }, botLifetime);
  });

  bot.on('error', (err) => {
    console.error(`❌ خطأ في ${bot.username}:`, err.code || err.message);

    // إذا كان خطأ اتصال، نؤجل جميع الجداول دقيقة ثم نعيد جدولة هذا البوت
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
  const delayMs = getIntervalForBot(botIndex);
  console.log(`📆 جدولة بوت #${botIndex} للدخول بعد ${Math.floor(delayMs / 1000)} ثانية.`);
  setTimeout(() => {
    if (!activeBots[botIndex]) {
      createBot(botIndex);
    }
  }, delayMs);
}

// بدء العملية: جدولة كل البوتات فورًا
(() => {
  console.log(`🛰️ تشغيل البوتات على إصدار Java Vanilla ${serverVersion}.`);
  for (let i = 0; i < maxBots; i++) {
    scheduleNextBot(i);
  }
})();
