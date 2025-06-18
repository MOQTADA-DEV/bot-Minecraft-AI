const mineflayer = require('mineflayer');

const serverAddress = 'MQTADA12.aternos.me';
const serverPort = 55865;

const maxBots = 20;
const botLifetime = 100 * 60 * 1000; // 100 دقيقة
const baseInterval = 300; // 5 دقائق
const maxInterval = 360; // 6 دقائق

let activeBots = new Array(maxBots).fill(null);
let scheduleDelays = new Array(maxBots).fill(0); // لكل بوت تأخير زمني إضافي بالدقائق (للأخطاء)

function getIntervalForBot(botNumber) {
  let interval = baseInterval + (botNumber % (maxInterval - baseInterval + 1));
  interval += scheduleDelays[botNumber]; // نضيف التأخير في حالة فشل سابق
  return interval * 1000;
}

function delayAllSchedulesOneMinute() {
  for (let i = 0; i < maxBots; i++) {
    scheduleDelays[i] += 1; // تأخير كل بوت دقيقة
  }
  console.log("📛 تم تأخير جميع الجداول دقيقة بسبب فشل اتصال.");
}

function createBot(botIndex) {
  const bot = mineflayer.createBot({
    host: serverAddress,
    port: serverPort,
    username: `Bot_${botIndex}_${Date.now()}`,
  });

  bot.on('spawn', () => {
    console.log(`✅ بوت ${bot.username} دخل السيرفر.`);
    activeBots[botIndex] = bot;

    setTimeout(() => {
      bot.quit();
      console.log(`⏱️ بوت ${bot.username} خرج بعد 100 دقيقة.`);
      activeBots[botIndex] = null;
      scheduleNextBot(botIndex); // بعد خروجه، أعد جدولة بوت جديد في نفس المكان
    }, botLifetime);
  });

  bot.on('error', (err) => {
    console.error(`❌ خطأ في البوت ${bot.username}:`, err.code || err.message);

    if (
      err.code === 'ECONNREFUSED' ||
      err.code === 'ECONNRESET' ||
      err.message.includes('read ECONNRESET')
    ) {
      console.log(`🔁 سيتم إعادة محاولة دخول البوت ${bot.username} بعد دقيقة.`);
      delayAllSchedulesOneMinute(); // نؤخر كل الجدول دقيقة
      setTimeout(() => scheduleNextBot(botIndex), 60 * 1000); // أعد المحاولة بعد دقيقة
    }
  });

  bot.on('end', () => {
    console.log(`🛑 البوت ${bot.username} أنهى الجلسة.`);
    activeBots[botIndex] = null;
  });
}

function scheduleNextBot(botIndex) {
  const interval = getIntervalForBot(botIndex);
  console.log(`📆 جدولة بوت رقم ${botIndex} للدخول بعد ${Math.floor(interval / 1000)} ثانية.`);

  setTimeout(() => {
    if (!activeBots[botIndex]) {
      createBot(botIndex);
    }
  }, interval);
}

// بدء أول 20 بوت (كل بوت في وقته المخصص)
for (let i = 0; i < maxBots; i++) {
  scheduleNextBot(i);
}
