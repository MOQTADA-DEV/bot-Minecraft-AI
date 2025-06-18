// =============================================================================
// بوت ماين كرافت متقدم باستخدام التعلم المعزز (Node.js)
// =============================================================================
// هذا الكود يطبق وكيل تعلم معزز (RL) متقدمًا باستخدام TensorFlow.js
// لبوت Mineflayer في لعبة Minecraft Java Edition.
// يركز البوت على البقاء، جمع الموارد، التقدم نحو أهداف كبرى (مثل النذر وتنين النهاية)،
// والتواصل الأساسي بين البوتات. كما يتضمن ميزات لإدارة عدة بوتات وحفظ
// نماذج التعلم الخاصة بها بين عمليات إعادة التشغيل.
//
// ملاحظات هامة:
// - يجب تشغيل هذا السكريبت في بيئة Node.js، وليس في متصفح الويب.
// - تطبيقات الخوارزميات واسعة النطاق مثل MuZero أو DreamerV3 تتجاوز بكثير
//   نطاق سكريبت Node.js واحد بسبب متطلباتها الحاسوبية والبنية التحتية الهائلة.
//   يوفر هذا الكود أساسًا قويًا لتعلم Q (DQN) لسلوكيات معقدة داخل اللعبة.
// - مكتبة `minecraft-data` قد لا تحتوي على بيانات دقيقة جدًا لإصدارات Minecraft
//   الحديثة جدًا (مثل 1.21.5). ستحاول تحميل أقرب البيانات المتاحة، عادةً الإصدار
//   الرئيسي (مثل 1.21).
//
// =============================================================================
// تعليمات الإعداد والتشغيل:
// =============================================================================
// 1. تأكد من تثبيت Node.js على جهازك (يمكنك تحميله من: https://nodejs.org/).
// 2. أنشئ مجلدًا جديدًا فارغًا لمشروعك (مثال: `minecraft_rl_bots`).
// 3. افتح واجهة سطر الأوامر (CMD أو PowerShell أو Terminal) وانتقل إلى مجلد مشروعك الجديد.
// 4. قم بتهيئة مشروع Node.js جديد (سيؤدي هذا إلى إنشاء ملف `package.json`):
//    `npm init -y`
// 5. قم بتثبيت المكتبات الضرورية (المدرجة في قسم `dependencies` في `package.json`):
//    `npm install mineflayer mineflayer-pathfinder @tensorflow/tfjs-node minecraft-data vec3`
// 6. احفظ كتلة الكود هذه بالكامل في ملف باسم `bot.js` داخل مجلد مشروعك.
//    (اسم الملف هذا يتوافق مع `main` و `start` في ملف `package.json` الخاص بك).
// 7. تأكد من تشغيل خادم Minecraft Java Edition على العنوان والمنفذ المحددين أدناه.
// 8. قم بتشغيل سكريبت مدير البوتات:
//    `node bot.js`
//    سيبدأ السكريبت الآن بالعدد الافتراضي من البوتات، ولن يطلب منك أي إدخال.
//
// =============================================================================


// =============================================================================
// I. استيراد المكتبات (Library Imports)
// =============================================================================
const mineflayer = require('mineflayer'); // المكتبة الأساسية لإنشاء بوتات ماين كرافت
const pathfinder = require('mineflayer-pathfinder').pathfinder; // لتمكين البوت من البحث عن المسارات والتنقل الذكي
const goals = require('mineflayer-pathfinder').goals; // أهداف التنقل الخاصة بـ pathfinder
const { GoalBlock, GoalNear, GoalXZ } = require('mineflayer-pathfinder').goals; // أنواع محددة من الأهداف
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js لواجهة برمجة تطبيقات Node.js (للتعلّم الآلي)
const vec3 = require('vec3'); // لأداء عمليات المتجهات ثلاثية الأبعاد (المواقع، الاتجاهات)
const fs = require('fs').promises; // وحدة نظام الملفات في Node.js (واجهة برمجة تطبيقات تستند إلى الوعود)
// لم تعد هناك حاجة لـ `readline` لأن العدد الافتراضي للبوتات يتم تعيينه مباشرة في الكود.

let mcData; // سيتم تهيئة بيانات لعبة Minecraft (الكتل، العناصر، إلخ) ديناميكيًا بعد اتصال البوت


// =============================================================================
// II. التكوين العام للبوتات والثوابت (Global Bot Configuration & Constants)
// =============================================================================
const SERVER_HOST = 'MQTADA12.aternos.me'; // عنوان خادم Minecraft
const SERVER_PORT = 55865;             // منفذ خادم Minecraft
const MINECRAFT_VERSION = '1.21.5';    // إصدار Minecraft للاتصال به

// مسار الملف لحفظ/تحميل ملفات تعريف البوتات (أسماء المستخدمين) لضمان الاستمرارية
const BOTS_CONFIG_FILE = './bots_config.json';
// دليل حفظ نماذج التعلم (الشبكات العصبية) لكل بوت على حدة
const MODELS_DIR = './bot_models';

// العدد الافتراضي للبوتات التي سيتم تشغيلها عند بدء السكريبت. يمكنك تغيير هذا الرقم مباشرة هنا.
const DEFAULT_BOT_COUNT = 10;


// =============================================================================
// III. تعريف إجراءات البوت (Defining Bot Actions)
// =============================================================================
// هذا الكائن يربط أسماء الإجراءات الوصفية بمعرفات رقمية فريدة.
// يمثل كل معرف رقمي عصبون إخراج في الشبكة العصبية لوكيل Q-learning.
const ACTIONS = {
    WALK_RANDOM: 0,                   // المشي عشوائيًا إلى موقع قريب
    ATTACK_NEAREST_HOSTILE: 1,        // مهاجمة أقرب كائن معادٍ
    FLEE_FROM_HOSTILE: 2,             // الهروب من أقرب كائن معادٍ
    BUILD_BASIC_SHELTER: 3,           // بناء ملجأ أساسي من التراب
    DIG_WOOD: 4,                      // تعدين أقرب جذع شجر
    CRAFT_WOODEN_PICKAXE: 5,          // صناعة فأس خشبي
    COLLECT_DROPPED_ITEMS: 6,         // جمع العناصر المتساقطة القريبة
    SLEEP: 7,                         // محاولة النوم في سرير (إذا كان الليل وسرير قريب)
    IDLE: 8,                          // عدم فعل أي شيء لفترة قصيرة
    MINE_STONE: 9,                    // تعدين حجر الكوبلستون (cobblestone)
    CRAFT_STONE_PICKAXE: 10,          // صناعة فأس حجري
    MINE_COAL: 11,                    // تعدين فحم الخام
    MINE_IRON: 12,                    // تعدين حديد الخام
    CRAFT_FURNACE: 13,                // صناعة فرن
    SMELT_IRON: 14,                   // صهر خام الحديد إلى سبائك
    CRAFT_IRON_PICKAXE: 15,           // صناعة فأس حديدي
    MINE_DIAMONDS: 16,                // محاولة تعدين الألماس (تتطلب فأس حديدي/ألماس)
    CRAFT_NETHER_PORTAL: 17,          // محاولة صناعة بوابة النذر (مفهومي)
    GO_TO_NETHER: 18,                 // محاولة دخول بُعد النذر (مفهومي)
    KILL_DRAGON: 19,                  // محاولة قتل تنين النهاية (مفهومي ومعقد للغاية)
    COMMUNICATE_REQUEST_WOOD: 20,     // محادثة: طلب الخشب من اللاعبين/البوتات الأخرى
    COMMUNICATE_OFFER_WOOD: 21,       // محادثة: عرض الخشب على اللاعبين/البوتات الأخرى
    COMMUNICATE_ENEMY_WARNING: 22,    // محادثة: التحذير من الكائنات المعادية القريبة
};
const ACTION_NAMES = Object.keys(ACTIONS); // مصفوفة بأسماء الإجراءات لتسهيل التسجيل والقراءة


// =============================================================================
// IV. فئة وكيل التعلّم المعزز (Q-Agent Class)
// =============================================================================
/**
 * تطبيق وكيل شبكة Q-العميقة (DQN) للتعلم المعزز.
 * يستخدم هذا الوكيل TensorFlow.js لبناء وتدريب شبكة عصبية تقارب دالة قيمة Q.
 * كل بوت سيكون له وكيل Q-Agent الخاص به، مما يسمح بالتعلم المستقل.
 */
class QAgent {
    /**
     * مُنشئ فئة QAgent.
     * @param {number} inputDim - أبعاد متجه الحالة المدخل (عدد الميزات التي تصف البيئة).
     * @param {number} outputDim - أبعاد متجه الإجراءات المخرجة (عدد الإجراءات الممكنة).
     * @param {string} modelId - معرف فريد لنموذج هذا الوكيل (عادةً اسم مستخدم البوت).
     * @param {object} [params={}] - معلمات اختيارية لوكيل التعلم المعزز.
     * @param {number} [params.learningRate=0.001] - معدل التعلم (معدل تعديل الأوزان أثناء التدريب).
     * @param {number} [params.discountFactor=0.99] - عامل الخصم (جاما)، يخصم المكافآت المستقبلية.
     * @param {number} [params.epsilon=1.0] - معدل الاستكشاف الأولي (لتطبيق سياسة إبسيلون-جشعة).
     * @param {number} [params.epsilonDecay=0.9995] - معدل اضمحلال إبسيلون بمرور الوقت (لتقليل الاستكشاف تدريجيًا).
     * @param {number} [params.minEpsilon=0.005] - الحد الأدنى لقيمة إبسيلون.
     * @param {number} [params.replayBufferSize=200000] - الحد الأقصى لحجم ذاكرة إعادة التجربة.
     * @param {number} [params.batchSize=64] - حجم الدفعة (batch) المستخدمة في كل خطوة تدريب.
     * @param {number} [params.targetUpdateFreq=200] - التكرار (بعدد خطوات التدريب) لتحديث الشبكة الهدف.
     */
    constructor(inputDim, outputDim, modelId, params = {}) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.modelId = modelId; // يُستخدم لتحديد مسار حفظ فريد للنموذج

        // المعلمات الفائقة (Hyperparameters) مع القيم الافتراضية
        this.learningRate = params.learningRate ?? 0.001;
        this.discountFactor = params.discountFactor ?? 0.99;
        this.epsilon = params.epsilon ?? 1.0;
        this.epsilonDecay = params.epsilonDecay ?? 0.9995;
        this.minEpsilon = params.minEpsilon ?? 0.005;

        this.replayBuffer = []; // ذاكرة إعادة التجربة لتخزين الخبرات
        this.replayBufferSize = params.replayBufferSize ?? 200000; // حجم الذاكرة
        this.batchSize = params.batchSize ?? 64; // حجم الدفعة للتدريب

        this.targetUpdateFreq = params.targetUpdateFreq ?? 200; // تكرار تحديث الشبكة الهدف
        this.updateCount = 0; // عداد لتتبع خطوات التدريب

        // تحديد مسار الحفظ الفريد لنموذج هذا البوت
        this.modelSavePath = `${MODELS_DIR}/${this.modelId}`;

        // بناء الشبكة العصبية الرئيسية (Q-Network) والشبكة الهدف (Target Q-Network)
        this.model = this._buildQNetwork();
        this.targetModel = this._buildQNetwork();
        // مزامنة أوزان الشبكة الهدف مع الشبكة الرئيسية في البداية
        this.targetModel.setWeights(this.model.getWeights());

        // محاولة تحميل أوزان النموذج المحفوظة مسبقًا للتعلم المستمر
        this._loadModel();
    }

    /**
     * تبني نموذج شبكة Q-العميقة (DQN) باستخدام واجهة TensorFlow.js التسلسلية.
     * هذه شبكة عصبية تغذية أمامية بسيطة. لتطبيق التعلم المعزز المتكرر (Recurrent RL)،
     * يجب تعديل هذه الدالة لتضمين طبقات مثل `tf.layers.lstm` أو `tf.layers.gru`
     * وتكييف شكل المدخلات لاستقبال تسلسلات من الحالات بدلاً من حالة واحدة.
     * @returns {tf.LayersModel} نموذج TensorFlow.js المترجم.
     * @private
     */
    _buildQNetwork() {
        const model = tf.sequential();
        // طبقة الإدخال: تستقبل متجه الحالة (عدد الميزات)
        model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [this.inputDim] }));
        // الطبقات المخفية: تقوم بمعالجة ميزات الإدخال
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        // طبقة الإخراج: تتنبأ بقيم Q لكل إجراء ممكن
        model.add(tf.layers.dense({ units: this.outputDim, activation: 'linear' })); // تنشيط خطي لقيم Q

        // ترجمة (تجميع) النموذج باستخدام محسّن (optimizer) ودالة خسارة (loss function)
        model.compile({
            optimizer: tf.train.adam(this.learningRate), // محسّن Adam شائع وفعال
            loss: 'meanSquaredError' // MSE (متوسط الخطأ التربيعي) هي دالة خسارة شائعة لـ Q-learning
        });
        return model;
    }

    /**
     * تحاول تحميل أوزان النموذج المدربة مسبقًا من نظام الملفات المحلي.
     * يتيح ذلك للبوت مواصلة التعلم من الجلسات السابقة.
     * @private
     */
    async _loadModel() {
        try {
            // التحقق مما إذا كان دليل النموذج موجودًا لهذا البوت المحدد
            await fs.access(this.modelSavePath);
            // تحميل أوزان النموذج من المسار المحدد
            this.model = await tf.loadLayersModel(`file://${this.modelSavePath}/model.json`);
            // مزامنة الشبكة الهدف مع النموذج الرئيسي الذي تم تحميله
            this.targetModel.setWeights(this.model.getWeights());
            console.log(`[RL_AGENT-${this.modelId}] تم تحميل النموذج بنجاح من ${this.modelSavePath}.`);
            // عند تحميل نموذج مدرب، من الشائع البدء بقيمة إبسيلون أقل
            this.epsilon = this.minEpsilon;
        } catch (e) {
            // إذا لم يكن الدليل أو ملف model.json موجودًا، فهذا أمر طبيعي لنموذج جديد
            if (e.code === 'ENOENT') {
                console.log(`[RL_AGENT-${this.modelId}] لم يتم العثور على نموذج موجود في ${this.modelSavePath}. بدء نموذج جديد.`);
            } else {
                // تسجيل أنواع أخرى من الأخطاء أثناء تحميل النموذج
                console.error(`[RL_AGENT-${this.modelId}] خطأ أثناء تحميل النموذج من ${this.modelSavePath}:`, e);
            }
        }
    }

    /**
     * يحفظ أوزان النموذج الحالي في نظام الملفات المحلي.
     * يجب استدعاء هذه الدالة بشكل دوري أو عند قطع اتصال البوت لضمان عدم فقدان التقدم.
     */
    async saveModel() {
        try {
            // إنشاء الدليل لنموذج هذا البوت إذا لم يكن موجودًا
            await fs.mkdir(this.modelSavePath, { recursive: true });
            // حفظ النموذج الرئيسي في المسار المحدد
            await this.model.save(`file://${this.modelSavePath}`);
            console.log(`[RL_AGENT-${this.modelId}] تم حفظ النموذج بنجاح في ${this.modelSavePath}.`);
        } catch (e) {
            console.error(`[RL_AGENT-${this.modelId}] فشل في حفظ النموذج في ${this.modelSavePath}:`, e);
        }
    }

    /**
     * يخزن تجربة واحدة (حالة، إجراء، مكافأة، حالة تالية، انتهاء) في ذاكرة إعادة التجربة.
     * هذه الذاكرة حاسمة للتدريب بالدفعات (batch training) وتثبيت عملية التعلم في DQN.
     * @param {Array<number>} state - الحالة الحالية الملاحظة للبيئة.
     * @param {number} action - الإجراء الذي اتخذه الوكيل.
     * @param {number} reward - المكافأة الفورية المستلمة من البيئة.
     * @param {Array<number>} nextState - الحالة الملاحظة بعد اتخاذ الإجراء.
     * @param {boolean} done - قيمة منطقية تشير إلى ما إذا كانت الحلقة قد انتهت.
     */
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push([state, action, reward, nextState, done]);
        // إذا تجاوزت الذاكرة حجمها الأقصى، يتم إزالة أقدم تجربة
        if (this.replayBuffer.length > this.replayBufferSize) {
            this.replayBuffer.shift();
        }
    }

    /**
     * يختار إجراءً بناءً على سياسة إبسيلون-جشعة (epsilon-greedy policy).
     * باحتمالية `epsilon`، يتم اختيار إجراء عشوائي (استكشاف).
     * وإلا، يتم اختيار الإجراء ذي أعلى قيمة Q متنبأ بها (استغلال).
     * @param {Array<number>} state - الحالة الحالية الملاحظة.
     * @returns {number} معرف الإجراء المختار.
     */
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
            // استكشاف: اختيار إجراء عشوائي من جميع الإجراءات الممكنة
            return Math.floor(Math.random() * this.outputDim);
        } else {
            // استغلال: التنبؤ بقيم Q لجميع الإجراءات في الحالة الحالية
            const stateTensor = tf.tensor2d([state]);
            const qValues = this.model.predict(stateTensor);
            // الحصول على معرف الإجراء ذي أعلى قيمة Q
            const action = qValues.argMax(-1).dataSync()[0];
            // تنظيف موترات TensorFlow من الذاكرة لمنع تسرب الذاكرة
            tf.dispose(stateTensor);
            tf.dispose(qValues);
            return action;
        }
    }

    /**
     * يدرب شبكة Q الرئيسية باستخدام دفعة مصغرة من التجارب المأخوذة
     * من ذاكرة إعادة التجربة. يطبق هذا جوهر خوارزمية DQN،
     * بحساب قيم Q المستهدفة باستخدام معادلة بيلمان وتحديث الشبكة.
     */
    async trainModel() {
        // التدريب فقط إذا كانت هناك تجارب كافية في الذاكرة لتشكيل دفعة
        if (this.replayBuffer.length < this.batchSize) {
            return;
        }

        // تحديث أوزان شبكة Q الهدف بشكل دوري لتثبيت التدريب
        this.updateCount++;
        if (this.updateCount % this.targetUpdateFreq === 0) {
            this.targetModel.setWeights(this.model.getWeights());
            console.log(`[RL_AGENT-${this.modelId}] تم تحديث أوزان الشبكة الهدف.`);
        }

        // أخذ عينة عشوائية من دفعة مصغرة من التجارب من ذاكرة إعادة التجربة
        const batch = this.replayBuffer
            .sort(() => 0.5 - Math.random()) // خلط الذاكرة عشوائياً
            .slice(0, this.batchSize);       // اختيار حجم الدفعة من التجارب المخلوطة

        // استخلاص مكونات التجارب
        const states = batch.map(e => e[0]);
        const actions = batch.map(e => e[1]);
        const rewards = batch.map(e => e[2]);
        const nextStates = batch.map(e => e[3]);
        const dones = batch.map(e => e[4]);

        // استخدام tf.tidy للتخلص تلقائيًا من الموترات المؤقتة، مما يمنع تسرب الذاكرة
        await tf.tidy(async () => {
            const statesTensor = tf.tensor2d(states);
            const nextStatesTensor = tf.tensor2d(nextStates);

            // التنبؤ بقيم Q للحالات الحالية باستخدام الشبكة الرئيسية
            const currentQValues = this.model.predict(statesTensor);
            // التنبؤ بقيم Q للحالات التالية باستخدام الشبكة الهدف (لأهداف أكثر استقرارًا)
            const nextQValuesTarget = this.targetModel.predict(nextStatesTensor);

            // تحويل قيم Q الحالية إلى مصفوفة JavaScript قابلة للتعديل
            const targetQValues = currentQValues.arraySync();

            // حساب قيمة Q المستهدفة لكل تجربة في الدفعة
            for (let i = 0; i < this.batchSize; i++) {
                const isDone = dones[i];
                const reward = rewards[i];
                const action = actions[i];

                if (isDone) {
                    // إذا انتهت الحلقة، فإن قيمة Q المستهدفة للإجراء المتخذ هي المكافأة الفورية فقط
                    targetQValues[i][action] = reward;
                } else {
                    // وإلا، يتم الحساب باستخدام معادلة بيلمان: R + gamma * max(Q_target(s', a'))
                    const maxNextQ = tf.max(nextQValuesTarget.slice([i, 0], [1, this.outputDim])).dataSync()[0];
                    targetQValues[i][action] = reward + this.discountFactor * maxNextQ;
                }
            }

            // تحويل قيم Q المستهدفة المحسوبة مرة أخرى إلى موتر TensorFlow
            const targetsTensor = tf.tensor2d(targetQValues);

            // تدريب شبكة Q الرئيسية للتنبؤ بقيم Q المستهدفة هذه.
            // هذا يضبط أوزان الشبكة لتقليل الفرق بين قيم Q المتنبأ بها وقيم Q المستهدفة.
            await this.model.fit(statesTensor, targetsTensor, {
                epochs: 1, // التدريب لـ epoch واحد لكل دفعة
                verbose: 0 // منع إخراج التدريب للحفاظ على نظافة وحدة التحكم
            });
        });

        // اضمحلال إبسيلون بمرور الوقت لتقليل الاستكشاف وتفضيل الاستغلال مع تعلم الوكيل
        this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    }
}


// =============================================================================
// V. تهيئة وكيل Q-Agent (Q-Agent Initialization)
// =============================================================================
// تعريف أبعاد متجه الحالة المدخل.
// يجب أن يتطابق هذا مع عدد القيم التي تعيدها الدالة `_getBotState()`.
const INPUT_NODES = 25;
// تعريف أبعاد متجه الإجراءات المخرجة (عدد الإجراءات الممكنة).
// يجب أن يتطابق هذا مع عدد الإجراءات المعرفة في الكائن `ACTIONS`.
const OUTPUT_NODES = Object.keys(ACTIONS).length;


// =============================================================================
// VI. فئة وكيل البوت (BotAgent Class)
// =============================================================================
/**
 * تدير مثيلاً واحدًا من بوت Mineflayer جنبًا إلى جنب مع وكيل Q-Agent المخصص له.
 * تتعامل مع دورة حياة البوت، مراقبة الحالة، تنفيذ الإجراءات، وتدريب التعلم المعزز.
 */
class BotAgent {
    /**
     * مُنشئ فئة BotAgent.
     * @param {string} username - اسم المستخدم الفريد لهذا البوت.
     * @param {object} [agentParams={}] - معلمات اختيارية لوكيل QAgent الخاص بهذا البوت.
     */
    constructor(username, agentParams = {}) {
        this.username = username;
        this.bot = null; // مثيل بوت Mineflayer
        this.qAgent = null; // مثيل QAgent لهذا البوت

        // متغيرات خاصة بالتعلم المعزز للحلقة الحالية (episode)
        this.lastState = null;
        this.lastAction = null;
        this.currentEpisodeReward = 0;
        this.consecutiveIdleActions = 0;
        this.lastDiedTime = 0; // طابع زمني لآخر موت للبوت (لميزة "recentlyDied" في الحالة)

        // تهيئة وكيل QAgent لهذا البوت.
        // INPUT_NODES و OUTPUT_NODES هما ثوابت عالمية لأبعاد QAgent.
        this.qAgent = new QAgent(INPUT_NODES, OUTPUT_NODES, this.username, agentParams);
    }

    /**
     * يقوم بتوصيل البوت بخادم Minecraft ويُعيّن مستمعي الأحداث.
     */
    connect() {
        this.bot = mineflayer.createBot({
            host: SERVER_HOST,
            port: SERVER_PORT,
            username: this.username,
            version: MINECRAFT_VERSION
        });

        // تحميل إضافة pathfinder للتنقل الذكي لهذا البوت
        this.bot.loadPlugin(pathfinder);

        // --- تسجيل مستمعي أحداث البوت ---
        this.bot.on('spawn', this._onSpawn.bind(this));      // عند اتصال البوت و ظهوره في العالم بنجاح
        this.bot.on('end', this._onEnd.bind(this));          // عند قطع اتصال البوت
        this.bot.on('error', this._onError.bind(this));      // عند حدوث خطأ
        this.bot.on('chat', this._onChat.bind(this));        // عند استقبال رسالة دردشة

        // تهيئة mcData (بيانات لعبة Minecraft) بشكل عام إذا لم يتم ذلك بعد.
        // هذا يضمن توفر mcData لجميع البوتات.
        if (!mcData) {
            this.bot.once('spawn', () => {
                mcData = require('minecraft-data')(this.bot.version);
                console.log(`[BOT-${this.username}][INFO] تم تهيئة بيانات ماينكرافت عالميًا للإصدار: ${this.bot.version}`);
            });
        }
    }

    /**
     * يفصل البوت عن الخادم.
     */
    disconnect() {
        if (this.bot && this.bot.client.socket.readyState === 'open') {
            this.bot.quit();
        }
    }

    /**
     * يتعامل مع حدث ظهور البوت. يعيد تعيين متغيرات الحلقة ويبدأ حلقة التفكير.
     * @private
     */
    async _onSpawn() {
        console.log(`[BOT-${this.username}] ظهر في العالم!`);
        this.bot.chat(`مرحباً! أنا ${this.username}, بوت ذكاء اصطناعي أتعلم البقاء والتطور.`);

        // إعادة تعيين المتغيرات الخاصة بالحلقة لتعلم حلقة جديدة
        this.lastState = null;
        this.lastAction = null;
        this.currentEpisodeReward = 0;
        this.consecutiveIdleActions = 0;

        // مسح أي فاصل زمني (interval) للتفكير موجود لمنع حلقات متعددة
        if (this.thinkInterval) {
            clearInterval(this.thinkInterval);
        }
        // بدء حلقة التفكير الرئيسية للبوت، باستدعاء _think() كل 1 ثانية
        this.thinkInterval = setInterval(this._think.bind(this), 1000);
        console.log(`[BOT-${this.username}][INFO] بدأت دورة التفكير للبوت.`);
    }

    /**
     * يتعامل مع حدث قطع اتصال البوت. يمسح حلقة التفكير ويحفظ النموذج.
     * @param {string} reason - سبب قطع الاتصال.
     * @private
     */
    async _onEnd(reason) {
        console.log(`[BOT-${this.username}] قطع الاتصال: ${reason}`);
        if (this.thinkInterval) {
            clearInterval(this.thinkInterval); // إيقاف حلقة التفكير
        }
        await this.qAgent.saveModel(); // حفظ أوزان النموذج المتعلمة عند قطع الاتصال
        console.log(`[BOT-${this.username}][INFO] تم حفظ النموذج بعد قطع الاتصال.`);
    }

    /**
     * يتعامل مع الأخطاء غير المعالجة التي تحدث في عملية البوت. يحفظ النموذج.
     * @param {Error} err - كائن الخطأ.
     * @private
     */
    async _onError(err) {
        console.error(`[BOT-${this.username}][ERROR] حدث خطأ غير معالج: ${err.message}`);
        // محاولة حفظ النموذج عند حدوث خطأ للحفاظ على تقدم التعلم
        await this.qAgent.saveModel();
    }

    /**
     * يتعامل مع رسائل الدردشة الواردة من اللاعبين أو البوتات الأخرى.
     * هذا هو الأساس للتواصل والتعاون بين البوتات.
     * @param {string} username - اسم المستخدم للمرسل.
     * @param {string} message - الرسالة المستلمة.
     * @private
     */
    async _onChat(username, message) {
        if (username === this.username) return; // تجاهل الرسائل المرسلة من البوت نفسه

        console.log(`[BOT-${this.username}][CHAT_RX] رسالة من ${username}: "${message}"`);

        // --- تحليل الرسائل البسيط لإجراءات التعاون ---
        // هذه استجابات تفاعلية أساسية. للتعلم المعزز متعدد الوكلاء (MARL) المتقدم،
        // ستتعلم البوتات بروتوكولات الاتصال والتنسيق الاستراتيجي.

        if (message.includes('Need wood!')) {
            const woodCount = this._getBotState()[8] * 64; // تحويل عدد الخشب إلى قيمة حقيقية
            if (woodCount > 10) {
                this.bot.chat(`@${username} ${this.username}: لدي ${woodCount.toFixed(0)} خشب! يمكنني مشاركة البعض في موقعي.`);
            } else {
                this.bot.chat(`@${username} ${this.username}: ليس لدي الكثير من الخشب، لكنني سأحاول إيجاد بعض.`);
            }
        } else if (message.includes('Have extra wood!')) {
            this.bot.chat(`@${username} ${this.username}: شكراً للعرض! قد آتي لأجمع إذا احتجت ذلك.`);
        } else if (message.includes('Warning! Hostile mob')) {
            console.log(`[BOT-${this.username}][COOP_ACTION] تلقيت تحذيرًا من عدو من ${username}. أولوية السلامة.`);
            const currentState = this._getBotState();
            // إذا كانت الصحة منخفضة، فكر جديًا في الهروب. وإلا، استعد للهجوم.
            if (currentState[0] < 0.5) {
                await this._executeBotAction(ACTIONS.FLEE_FROM_HOSTILE);
            } else {
                // إذا لم يكن على دراية بوجود عدو بالفعل، حاول الهجوم
                if (currentState[4] === 0) {
                    await this._executeBotAction(ACTIONS.ATTACK_NEAREST_HOSTILE);
                }
            }
        }
    }

    /**
     * يجمع ويوحد الحالة الحالية للبوت وبيئته في مصفوفة رقمية مناسبة كمدخل للشبكة العصبية.
     * تحدد هذه الدالة "مساحة الملاحظة" (observation space) للوكيل،
     * بما في ذلك الموارد، الصحة، وجود الأعداء، وتقدم اللعبة.
     * @returns {number[]} مصفوفة من القيم الرقمية الموحدة التي تمثل حالة البوت.
     * @private
     */
    _getBotState() {
        // التأكد من توفر mcData (يجب تهيئته عند أول ظهور للبوت)
        if (!mcData) {
            console.warn(`[BOT-${this.username}][STATE_ERR] لم يتم تهيئة mcData بعد. إرجاع حالة افتراضية.`);
            return new Array(INPUT_NODES).fill(0); // إرجاع مصفوفة أصفار إذا كان mcData مفقودًا
        }
        if (!this.bot || !this.bot.entity) {
            console.warn(`[BOT-${this.username}][STATE_ERR] كائن البوت أو الكيان غير متاح. إرجاع حالة افتراضية.`);
            return new Array(INPUT_NODES).fill(0);
        }

        // --- إحصائيات البقاء الأساسية (موحدة من 0-1) ---
        const health = this.bot.health / 20;
        const food = this.bot.food / 20;
        const inWater = this.bot.entity.isInWater ? 1 : 0;
        const isDay = this.bot.time.timeOfDay < 13000 ? 1 : 0; // اليوم التقريبي (0-24000 تكة)

        // --- وجود الأعداء (موحدة) ---
        const nearestHostile = this.bot.nearestEntity(entity =>
            entity.type === 'mob' && entity.mobType && entity.mobType.isHostile &&
            entity.position.distanceTo(this.bot.entity.position) < 30
        );
        const hasEnemyNearby = nearestHostile ? 1 : 0;
        const enemyDistance = nearestHostile ? nearestHostile.position.distanceTo(this.bot.entity.position) / 30 : 0; // 0-1 إذا كان ضمن 30 بلوك

        // --- المخزون والموارد (موحدة حسب الحد الأقصى للمكدس أو الكميات الشائعة) ---
        const hasFoodInInventory = this.bot.inventory.items().some(item => mcData.foodsArray.some(food => food.id === item.type)) ? 1 : 0;
        const woodCount = this.bot.inventory.count(mcData.itemsByName.oak_log.id) / 64; // أقصى مكدس 64
        const hasWoodenPickaxe = this.bot.inventory.items().some(item => item.name === 'wooden_pickaxe') ? 1 : 0;
        const stoneCount = this.bot.inventory.count(mcData.itemsByName.cobblestone.id) / 64;
        const hasStonePickaxe = this.bot.inventory.items().some(item => item.name === 'stone_pickaxe') ? 1 : 0;
        const coalCount = this.bot.inventory.count(mcData.itemsByName.coal.id) / 64;
        const ironOreCount = this.bot.inventory.count(mcData.itemsByName.iron_ore.id) / 64;
        const ironIngotCount = this.bot.inventory.count(mcData.itemsByName.iron_ingot.id) / 64;
        const hasIronPickaxe = this.bot.inventory.items().some(item => item.name === 'iron_pickaxe') ? 1 : 0;
        const obsidianCount = this.bot.inventory.count(mcData.itemsByName.obsidian.id) / 64;
        const hasDiamondPickaxe = this.bot.inventory.items().some(item => item.name === 'diamond_pickaxe') ? 1 : 0;
        const enderEyesCount = this.bot.inventory.count(mcData.itemsByName.ender_eye.id) / 12; // نحتاج 12 لبوابة النهاية

        // --- تقدم اللعبة والموقع (علامات ثنائية) ---
        const netherPortalExistsNearby = !!this.bot.findBlock({
            matching: mcData.blocksByName.nether_portal.id,
            maxDistance: 100 // التحقق من منطقة أوسع للبوابات
        });
        const inNether = this.bot.entity.dimension === 'minecraft:the_nether' ? 1 : 0;
        const inEnd = this.bot.entity.dimension === 'minecraft:the_end' ? 1 : 0;

        // --- "الذاكرة" الأساسية / الحالة التاريخية (مفهومي للتعلم المعزز المتكرر) ---
        // علامة بسيطة للأحداث الأخيرة. شبكة RNN حقيقية ستقوم بمعالجة تسلسلات من الحالات.
        const recentlyDied = this.lastDiedTime && (Date.now() - this.lastDiedTime < 10000) ? 1 : 0; // مات في آخر 10 ثوانٍ

        // --- حالة التعاون متعدد الوكلاء ---
        const nearestOtherPlayer = this.bot.nearestEntity(entity => entity.type === 'player' && entity.username !== this.username);
        const hasOtherPlayerNearby = nearestOtherPlayer ? 1 : 0;
        const otherPlayerDistance = nearestOtherPlayer ? nearestOtherPlayer.position.distanceTo(this.bot.entity.position) / 60 : 0; // موحدة بـ 60 بلوك

        return [
            health, food, inWater, isDay, hasEnemyNearby, enemyDistance,
            hasFoodInInventory, woodCount, hasWoodenPickaxe, stoneCount,
            hasStonePickaxe, coalCount, ironOreCount, ironIngotCount,
            hasIronPickaxe, obsidianCount, hasDiamondPickaxe, enderEyesCount,
            netherPortalExistsNearby, inNether, inEnd, // هذه القيم بالفعل 0/1 من التحويل المنطقي
            recentlyDied, hasOtherPlayerNearby, otherPlayerDistance
        ];
    }

    /**
     * ينفذ إجراء بوت محدد بناءً على معرفه. تحتوي هذه الدالة على المنطق المعقد
     * لإجراءات متعددة الخطوات وتتفاعل مع Mineflayer API.
     * تم تحسينها لتوفير رسائل خطأ أكثر تفصيلاً.
     * @param {number} actionId - معرف الإجراء المراد تنفيذه.
     * @returns {Promise<{success: boolean, actionReward: number}>} كائن يشير إلى النجاح والمكافأة الفورية.
     * @private
     */
    async _executeBotAction(actionId) {
        console.log(`[BOT-${this.username}][ACTION_EXEC] محاولة تنفيذ: ${ACTION_NAMES[actionId]}`);
        let success = false;
        let actionReward = 0; // المكافأة الفورية لهذا الإجراء المحدد

        // التحقق من توفر كائن البوت قبل محاولة تنفيذ أي إجراءات
        if (!this.bot || !this.bot.entity) {
            console.error(`[BOT-${this.username}][ACTION_ERR] كائن البوت غير متاح لتنفيذ الإجراء ${ACTION_NAMES[actionId]}.`);
            return { success: false, actionReward: -1.0 };
        }
        if (!mcData) {
            console.error(`[BOT-${this.username}][ACTION_ERR] بيانات Minecraft (mcData) غير مهيأة لتنفيذ الإجراء ${ACTION_NAMES[actionId]}.`);
            return { success: false, actionReward: -1.0 };
        }


        try {
            switch (actionId) {
                case ACTIONS.WALK_RANDOM: {
                    const randomOffset = Math.random() * 20 - 10;
                    const randomX = this.bot.entity.position.x + randomOffset;
                    const randomZ = this.bot.entity.position.z + randomOffset;
                    const randomGoal = new GoalXZ(randomX, randomZ);
                    try {
                        await this.bot.pathfinder.goto(randomGoal);
                        actionReward = 0.1;
                        success = true;
                        console.log(`[BOT-${this.username}][INFO] نجاح المشي العشوائي إلى X:${randomX.toFixed(0)}, Z:${randomZ.toFixed(0)}.`);
                    } catch (err) {
                        console.log(`[BOT-${this.username}][WARN] فشل المشي العشوائي: ${err.message}`);
                        actionReward = -0.1;
                    }
                    break;
                }

                case ACTIONS.ATTACK_NEAREST_HOSTILE: {
                    const nearestHostile = this.bot.nearestEntity(entity =>
                        entity.type === 'mob' && entity.mobType && entity.mobType.isHostile &&
                        entity.position.distanceTo(this.bot.entity.position) < 5
                    );
                    if (nearestHostile) {
                        this.bot.setControlState('forward', true);
                        await this.bot.attack(nearestHostile);
                        this.bot.setControlState('forward', false);
                        actionReward = 0.5;
                        success = true;
                        console.log(`[BOT-${this.username}][INFO] هاجمت ${nearestHostile.displayName} بنجاح.`);
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد كائن معادٍ قريب للهجوم.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                case ACTIONS.FLEE_FROM_HOSTILE: {
                    const nearestHostile = this.bot.nearestEntity(entity =>
                        entity.type === 'mob' && entity.mobType && entity.mobType.isHostile
                    );
                    if (nearestHostile) {
                        const fleeVector = this.bot.entity.position.minus(nearestHostile.position).normalize().scaled(15);
                        const fleeGoal = new GoalNear(
                            this.bot.entity.position.x + fleeVector.x,
                            this.bot.entity.position.y,
                            this.bot.entity.position.z + fleeVector.z, 5
                        );
                        try {
                            await this.bot.pathfinder.goto(fleeGoal);
                            actionReward = 0.8;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] نجاح الهروب من ${nearestHostile.displayName}.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل محاولة الهروب: ${err.message}`);
                            actionReward = -0.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد كائن معادٍ للهروب منه.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                case ACTIONS.BUILD_BASIC_SHELTER: {
                    const dirtId = mcData.blocksByName.dirt.id;
                    const hasDirt = this.bot.inventory.count(dirtId);

                    if (hasDirt >= 16) {
                        try {
                            const dirtItem = this.bot.inventory.items().find(item => item.type === dirtId);
                            if (!dirtItem) {
                                console.log(`[BOT-${this.username}][WARN] لا يوجد عنصر تراب في المخزون على الرغم من العدد الكافي.`);
                                actionReward = -0.2;
                                break;
                            }
                            await this.bot.setQuickBarSlot(dirtItem.slot);
                            const currentPos = this.bot.entity.position.floor();

                            // بناء قاعدة (أرضية) 3x3
                            for (let x = -1; x <= 1; x++) {
                                for (let z = -1; z <= 1; z++) {
                                    const targetPos = currentPos.offset(x, -1, z);
                                    const referenceBlock = this.bot.blockAt(targetPos);
                                    if (referenceBlock && referenceBlock.type !== dirtId && referenceBlock.type !== 0 && this.bot.canPlaceBlock(referenceBlock)) {
                                        await this.bot.placeBlock(referenceBlock, vec3(0, 1, 0));
                                        actionReward += 0.1;
                                        await this.bot.waitForTicks(5);
                                    }
                                }
                            }
                            // بناء 3 طبقات من الجدران (صندوق بسيط)
                            for (let y = 0; y < 3; y++) {
                                for (let x = -1; x <= 1; x++) {
                                    for (let z = -1; z <= 1; z++) {
                                        // وضع الجدران الخارجية فقط، وليس داخل مساحة البوت
                                        if (Math.abs(x) === 1 || Math.abs(z) === 1) {
                                            const targetPos = currentPos.offset(x, y, z);
                                            const referenceBlock = this.bot.blockAt(targetPos.offset(0, -1, 0));
                                            if (referenceBlock && referenceBlock.type !== dirtId && this.bot.canPlaceBlock(referenceBlock)) {
                                                await this.bot.placeBlock(referenceBlock, targetPos.minus(referenceBlock.position));
                                                actionReward += 0.15;
                                                await this.bot.waitForTicks(5);
                                            }
                                        }
                                    }
                                }
                            }
                            actionReward += 5.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم بناء ملجأ ترابي أساسي بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل البناء: ${err.message}`);
                            actionReward = -1.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد تراب كافٍ للبناء (تحتاج 16).`);
                        actionReward = -0.2;
                    }
                    break;
                }

                case ACTIONS.DIG_WOOD: {
                    const woodBlock = this.bot.findBlock({
                        matching: block => block.name.includes('log') || block.name.includes('wood'),
                        maxDistance: 16
                    });
                    if (woodBlock && this.bot.canDigBlock(woodBlock)) {
                        try {
                            await this.bot.dig(woodBlock);
                            actionReward = 1.5;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم تعدين الخشب بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل تعدين الخشب: ${err.message}`);
                            actionReward = -0.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد جذع شجر قريب لتعدينه.`);
                        actionReward = -0.2;
                    }
                    break;
                }

                case ACTIONS.CRAFT_WOODEN_PICKAXE: {
                    const woodenPickaxeRecipe = this.bot.recipesFor(mcData.itemsByName.wooden_pickaxe.id).find(r => true);
                    if (woodenPickaxeRecipe) {
                        try {
                            await this.bot.craft(woodenPickaxeRecipe, 1);
                            actionReward = 2.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم صناعة فأس خشبي بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل صناعة فأس خشبي: ${err.message}`);
                            actionReward = -1.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا توجد وصفة للفأس الخشبي أو المواد مفقودة.`);
                        actionReward = -0.5;
                    }
                    break;
                }

                case ACTIONS.COLLECT_DROPPED_ITEMS: {
                    const itemDrop = this.bot.nearestEntity(entity => entity.type === 'item' && entity.position.distanceTo(this.bot.entity.position) < 10);
                    if (itemDrop) {
                        try {
                            await this.bot.collect(itemDrop);
                            actionReward = 0.7;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم جمع العنصر المتساقط بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل جمع العنصر: ${err.message}`);
                            actionReward = -0.3;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا توجد عناصر متساقطة قريبة لجمعها.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                case ACTIONS.SLEEP: {
                    const bed = this.bot.findBlock({
                        matching: block => block.name.includes('bed'),
                        maxDistance: 16,
                        count: 1
                    });
                    if (bed && !this._getBotState()[3]) { // إذا كان الليل
                        try {
                            await this.bot.sleep(bed);
                            actionReward = 1.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] البوت نائم.`);
                            setTimeout(async () => {
                                await this.bot.wake();
                                console.log(`[BOT-${this.username}][INFO] البوت استيقظ.`);
                            }, 5000); // يستيقظ بعد 5 ثوانٍ (محاكاة مرور الليل)
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل النوم: ${err.message}`);
                            actionReward = -0.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد سرير قريب أو ليس وقت الليل.`);
                        actionReward = -0.2;
                    }
                    break;
                }

                case ACTIONS.IDLE: {
                    actionReward = 0.05; // مكافأة صغيرة لمنع التقييم السلبي المستمر
                    success = true;
                    console.log(`[BOT-${this.username}][INFO] البوت في وضع الخمول.`);
                    break;
                }

                case ACTIONS.MINE_STONE: {
                    const stoneBlock = this.bot.findBlock({
                        matching: mcData.blocksByName.stone.id,
                        maxDistance: 16
                    });
                    const hasWoodenPick = this._getBotState()[9]; // التحقق من وجود فأس خشبي
                    if (stoneBlock && this.bot.canDigBlock(stoneBlock) && hasWoodenPick) {
                        try {
                            await this.bot.dig(stoneBlock);
                            actionReward = 1.8;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم تعدين الحجر بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل تعدين الحجر: ${err.message}`);
                            actionReward = -0.6;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد حجر قريب أو فأس خشبي مفقود.`);
                        actionReward = -0.3;
                    }
                    break;
                }

                case ACTIONS.CRAFT_STONE_PICKAXE: {
                    const stonePickaxeRecipe = this.bot.recipesFor(mcData.itemsByName.stone_pickaxe.id).find(r => true);
                    const hasEnoughStone = this._getBotState()[10] * 64 >= 3; // عدد الكوبلستون بعد إلغاء التوحيد
                    if (stonePickaxeRecipe && hasEnoughStone) {
                        try {
                            await this.bot.craft(stonePickaxeRecipe, 1);
                            actionReward = 3.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم صناعة فأس حجري بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل صناعة فأس حجري: ${err.message}`);
                            actionReward = -1.2;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا توجد وصفة للفأس الحجري أو المواد مفقودة (3 كوبلستون).`);
                        actionReward = -0.6;
                    }
                    break;
                }

                case ACTIONS.MINE_COAL: {
                    const coalOre = this.bot.findBlock({
                        matching: mcData.blocksByName.coal_ore.id,
                        maxDistance: 32
                    });
                    const hasStoneOrBetterPick = this._getBotState()[11] || this._getBotState()[15] || this._getBotState()[17];
                    if (coalOre && this.bot.canDigBlock(coalOre) && hasStoneOrBetterPick) {
                        try {
                            await this.bot.dig(coalOre);
                            actionReward = 2.5;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم تعدين الفحم بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل تعدين الفحم: ${err.message}`);
                            actionReward = -0.8;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد فحم خام قريب أو فأس مناسب مفقود.`);
                        actionReward = -0.4;
                    }
                    break;
                }

                case ACTIONS.MINE_IRON: {
                    const ironOre = this.bot.findBlock({
                        matching: mcData.blocksByName.iron_ore.id,
                        maxDistance: 32
                    });
                    const hasStoneOrBetterPick = this._getBotState()[11] || this._getBotState()[15] || this._getBotState()[17];
                    if (ironOre && this.bot.canDigBlock(ironOre) && hasStoneOrBetterPick) {
                        try {
                            await this.bot.dig(ironOre);
                            actionReward = 4.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم تعدين الحديد بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل تعدين الحديد: ${err.message}`);
                            actionReward = -1.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد خام حديد قريب أو فأس مناسب مفقود.`);
                        actionReward = -0.5;
                    }
                    break;
                }

                case ACTIONS.CRAFT_FURNACE: {
                    const furnaceRecipe = this.bot.recipesFor(mcData.itemsByName.furnace.id).find(r => true);
                    const hasEnoughCobblestone = this._getBotState()[10] * 64 >= 8;
                    if (furnaceRecipe && hasEnoughCobblestone) {
                        try {
                            await this.bot.craft(furnaceRecipe, 1);
                            actionReward = 3.5;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم صناعة فرن بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل صناعة فرن: ${err.message}`);
                            actionReward = -1.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا توجد وصفة للفرن أو المواد مفقودة (8 كوبلستون).`);
                        actionReward = -0.7;
                    }
                    break;
                }

                case ACTIONS.SMELT_IRON: {
                    const hasFurnaceInInventory = this.bot.inventory.items().some(item => item.name === 'furnace');
                    const hasCoal = this._getBotState()[12] * 64 >= 1;
                    const hasRawIron = this._getBotState()[13] * 64 >= 1;

                    if (hasFurnaceInInventory && hasCoal && hasRawIron) {
                        console.log(`[BOT-${this.username}][INFO] محاولة صهر الحديد (مفهومي).`);
                        actionReward = 5.0;
                        success = true;
                        // التتطبيق الفعلي للصهر معقد ويتطلب وضع الفرن، التفاعل مع مخزونه، الانتظار، ثم أخذ السبائك.
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يمكن صهر الحديد: فرن أو فحم أو خام حديد مفقود.`);
                        actionReward = -0.8;
                    }
                    break;
                }

                case ACTIONS.CRAFT_IRON_PICKAXE: {
                    const ironPickaxeRecipe = this.bot.recipesFor(mcData.itemsByName.iron_pickaxe.id).find(r => true);
                    const hasEnoughIronIngots = this._getBotState()[14] * 64 >= 3;
                    if (ironPickaxeRecipe && hasEnoughIronIngots) {
                        try {
                            await this.bot.craft(ironPickaxeRecipe, 1);
                            actionReward = 7.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] تم صناعة فأس حديدي بنجاح.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل صناعة فأس حديدي: ${err.message}`);
                            actionReward = -2.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا توجد وصفة للفأس الحديدي أو المواد مفقودة (3 سبائك حديد).`);
                        actionReward = -1.0;
                    }
                    break;
                }

                case ACTIONS.MINE_DIAMONDS: {
                    const hasIronOrDiamondPick = this._getBotState()[15] || this._getBotState()[17];
                    if (hasIronOrDiamondPick) {
                        const diamondOre = this.bot.findBlock({
                            matching: mcData.blocksByName.diamond_ore.id,
                            maxDistance: 32
                        });
                        if (diamondOre && this.bot.canDigBlock(diamondOre)) {
                            console.log(`[BOT-${this.username}][INFO] تم العثور على خام ألماس، محاولة تعدينه.`);
                            try {
                                await this.bot.dig(diamondOre);
                                actionReward = 15.0;
                                success = true;
                                console.log(`[BOT-${this.username}][INFO] تم تعدين الألماس بنجاح.`);
                            } catch (err) {
                                console.log(`[BOT-${this.username}][WARN] فشل تعدين الألماس: ${err.message}`);
                                actionReward = -3.0;
                            }
                        } else {
                            console.log(`[BOT-${this.username}][WARN] لا يوجد خام ألماس قريب.`);
                            actionReward = -1.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد فأس مناسب (حديدي أو ألماس) لتعدين الألماس.`);
                        actionReward = -1.5;
                    }
                    break;
                }

                case ACTIONS.CRAFT_NETHER_PORTAL: {
                    const obsidianCount = this._getBotState()[16] * 64;
                    const hasFlintAndSteel = this.bot.inventory.items().some(item => item.name === 'flint_and_steel');

                    if (obsidianCount >= 14 && hasFlintAndSteel) {
                        console.log(`[BOT-${this.username}][INFO] محاولة صناعة بوابة النذر (مفهومي).`);
                        actionReward = 50.0;
                        success = true;
                        // منطق بناء البوابة الفعلي معقد للغاية، ويتضمن وضع كتل دقيق.
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد أوبسيديان كافٍ (تحتاج 14) أو لا يوجد فلينت وستيل لصناعة البوابة.`);
                        actionReward = -2.0;
                    }
                    break;
                }

                case ACTIONS.GO_TO_NETHER: {
                    const netherPortalBlock = this.bot.findBlock({
                        matching: mcData.blocksByName.nether_portal.id,
                        maxDistance: 10
                    });
                    if (netherPortalBlock) {
                        console.log(`[BOT-${this.username}][INFO] تم العثور على بوابة النذر، محاولة الدخول.`);
                        try {
                            await this.bot.pathfinder.goto(new GoalBlock(netherPortalBlock.position.x, netherPortalBlock.position.y, netherPortalBlock.position.z));
                            actionReward = 100.0;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] دخلت البوت النذر بنجاح (مفهومي).`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][WARN] فشل الانتقال إلى بوابة النذر: ${err.message}`);
                            actionReward = -10.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لم يتم العثور على بوابة نذر قريبة للدخول.`);
                        actionReward = -2.0;
                    }
                    break;
                }

                case ACTIONS.KILL_DRAGON: {
                    console.log(`[BOT-${this.username}][INFO] محاولة قتل تنين النهاية (إجراء مفهومي).`);
                    if (this._getBotState()[20] === 1) { // إذا كان البوت في بُعد النهاية
                        actionReward = 1000.0;
                        success = true;
                        console.log(`[BOT-${this.username}][INFO] مكان مؤقت: سيقوم البوت الآن بتنفيذ منطق قتال تنين النهاية المعقد.`);
                    } else {
                        console.log(`[BOT-${this.username}][WARN] ليس في بُعد النهاية لقتل التنين.`);
                        actionReward = -5.0;
                    }
                    break;
                }

                case ACTIONS.COMMUNICATE_REQUEST_WOOD: {
                    const currentWood = this._getBotState()[8] * 64;
                    const hasOtherPlayerNearby = this._getBotState()[23];

                    if (currentWood < 10 && hasOtherPlayerNearby) {
                        const nearestPlayer = this.bot.nearestEntity(entity => entity.type === 'player' && entity.username !== this.username);
                        if (nearestPlayer) {
                            this.bot.chat(`@${nearestPlayer.username} ${this.username}: أحتاج للخشب! هل يمكنك المساعدة؟`);
                            actionReward = 0.2;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] أرسلت طلب خشب إلى ${nearestPlayer.username}.`);
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا أحتاج للخشب أو لا يوجد لاعبون آخرون قريبون لطلب المساعدة.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                case ACTIONS.COMMUNICATE_OFFER_WOOD: {
                    const currentWood = this._getBotState()[8] * 64;
                    const hasOtherPlayerNearby = this._getBotState()[23];

                    if (currentWood > 30 && hasOtherPlayerNearby) {
                        const nearestPlayer = this.bot.nearestEntity(entity => entity.type === 'player' && entity.username !== this.username);
                        if (nearestPlayer) {
                            this.bot.chat(`@${nearestPlayer.username} ${this.username}: لدي خشب إضافي إذا احتجت!`);
                            actionReward = 0.2;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] عرضت خشبًا على ${nearestPlayer.username}.`);
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] ليس لدي خشب كافٍ للعرض أو لا يوجد لاعبون آخرون قريبون.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                case ACTIONS.COMMUNICATE_ENEMY_WARNING: {
                    const hasEnemy = this._getBotState()[4];
                    const hasOtherPlayerNearby = this._getBotState()[23];

                    if (hasEnemy && hasOtherPlayerNearby) {
                        const nearestPlayer = this.bot.nearestEntity(entity => entity.type === 'player' && entity.username !== this.username);
                        const enemy = this.bot.nearestEntity(entity => entity.type === 'mob' && entity.mobType && entity.mobType.isHostile);
                        if (nearestPlayer && enemy) {
                            this.bot.chat(`@${nearestPlayer.username} ${this.username}: تحذير! كائن معادٍ (${enemy.displayName}) في موقعي: ${enemy.position.x.toFixed(0)}, ${enemy.position.y.toFixed(0)}, ${enemy.position.z.toFixed(0)}!`);
                            actionReward = 0.5;
                            success = true;
                            console.log(`[BOT-${this.username}][INFO] حذرت ${nearestPlayer.username} من العدو.`);
                        }
                    } else {
                        console.log(`[BOT-${this.username}][WARN] لا يوجد عدو للتحذير منه أو لا يوجد لاعبون آخرون قريبون.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                default: {
                    console.log(`[BOT-${this.username}][ERROR] معرف إجراء غير معروف: ${actionId}`);
                    actionReward = -0.05;
                    break;
                }
            }
        } catch (e) {
            console.error(`[BOT-${this.username}][EXCEPTION] استثناء أثناء تنفيذ الإجراء ${ACTION_NAMES[actionId]}: ${e.message}`);
            actionReward = -0.7; // عقوبة شديدة للأخطاء غير المعالجة أثناء الإجراء
        }

        return { success, actionReward };
    }


    /**
     * يحسب المكافأة التي يتلقاها البوت بناءً على الانتقال من حالة قديمة إلى حالة جديدة بعد اتخاذ إجراء.
     * هذا أمر حاسم لتوجيه عملية تعلم وكيل التعلم المعزز.
     * يتم استخدام تشكيل المكافأة (Reward Shaping) لتشجيع السلوكيات المرغوبة وتثبيط غير المرغوبة.
     * @param {Array<number>} oldState - حالة البوت قبل الإجراء.
     * @param {Array<number>} newState - حالة البوت بعد الإجراء.
     * @param {boolean} actionSuccess - يشير إلى ما إذا كان الإجراء الذي تم تنفيذه ناجحًا.
     * @returns {number} قيمة المكافأة المحسوبة.
     * @private
     */
    _calculateReward(oldState, newState, actionSuccess) {
        let reward = 0;

        // --- البقاء الأساسي والصحة ---
        reward += 0.01; // مكافأة إيجابية صغيرة لمجرد البقاء على قيد الحياة في كل تكة
        reward += (newState[0] - oldState[0]) * 10; // مكافأة كبيرة لزيادة الصحة، وعقوبة كبيرة لفقدان الصحة
        reward += (newState[1] - oldState[1]) * 5;  // مكافأة لزيادة الغذاء، وعقوبة لفقدانه
        if (newState[0] < 0.2 && newState[0] < oldState[0]) { reward -= 15; } // عقوبة شديدة لانخفاض الصحة بشكل حرج
        if (newState[1] < 0.2 && newState[1] < oldState[1]) { reward -= 10; } // عقوبة شديدة لانخفاض الغذاء بشكل حرج
        if (newState[21] === 1 && oldState[21] === 0) { reward -= 50; } // عقوبة كبيرة للموت (علامة recentlyDied)

        // --- التفاعل مع الأعداء ---
        if (newState[4] === 0 && oldState[4] === 1) { reward += 20; } // مكافأة كبيرة للهروب/القضاء على عدو بنجاح
        else if (newState[4] === 1 && oldState[4] === 1 && newState[5] < oldState[5]) { reward -= 3; } // عقوبة إذا اقترب العدو
        else if (newState[4] === 1 && oldState[4] === 0) { reward -= 7; } // عقوبة لظهور عدو جديد قريب

        // --- تقدم جمع الموارد (مكافآت حسب المستويات) ---
        // خشب
        if (newState[8] > oldState[8]) { reward += (newState[8] - oldState[8]) * 5; }
        // حجر/كوبلستون
        if (newState[9] > oldState[9]) { reward += (newState[9] - oldState[9]) * 7; }
        // فحم
        if (newState[11] > oldState[11]) { reward += (newState[11] - oldState[11]) * 8; }
        // خام الحديد
        if (newState[12] > oldState[12]) { reward += (newState[12] - oldState[12]) * 15; }
        // سبائك الحديد (من الصهر)
        if (newState[13] > oldState[13]) { reward += (newState[13] - oldState[13]) * 20; }
        // أوبسيديان
        if (newState[16] > oldState[16]) { reward += (newState[16] - oldState[16]) * 30; }
        // عيون الإندر
        if (newState[18] > oldState[18]) { reward += (newState[18] - oldState[18]) * 40; }

        // --- تقدم الأدوات/الصناعة (مكافآت الإنجاز) ---
        if (newState[9] === 1 && oldState[9] === 0) { reward += 10; } // صناعة فأس خشبي
        if (newState[11] === 1 && oldState[11] === 0) { reward += 20; } // صناعة فأس حجري
        if (newState[15] === 1 && oldState[15] === 0) { reward += 40; } // صناعة فأس حديدي
        if (newState[17] === 1 && oldState[17] === 0) { reward += 80; } // صناعة فأس ألماس

        // --- أهداف التقدم الرئيسية في اللعبة (مكافآت عالية جدًا) ---
        if (newState[18] === 1 && oldState[18] === 0) { reward += 200; } // بناء بوابة النذر
        if (newState[19] === 1 && oldState[19] === 0) { reward += 500; } // دخول النذر
        if (newState[20] === 1 && oldState[20] === 0) { reward += 1000; } // دخول النهاية
        // مكافأة ضخمة افتراضية لقتل تنين النهاية (سيتم تفعيلها بحدث داخل اللعبة)
        // if (DRAGON_KILLED_GLOBAL_EVENT) { reward += 5000; }

        // --- مكافآت التعاون متعدد الوكلاء ---
        if (newState[23] === 1 && oldState[23] === 0) { reward += 0.5; }

        // --- عقوبة الإجراءات الفاشلة ---
        if (!actionSuccess) {
            reward -= 1.0;
        }

        return reward;
    }

    /**
     * حلقة التفكير الرئيسية لوكيل البوت المحدد هذا.
     * يلاحظ البيئة، يتخذ القرارات، ينفذ الإجراءات، ويتعلم.
     * يتم استدعاء هذه الدالة بشكل متكرر بواسطة `setInterval`.
     * @private
     */
    async _think() {
        // التأكد من تحميل بيانات Minecraft قبل المتابعة
        if (!mcData) {
            console.log(`[BOT-${this.username}][THINK] انتظار تهيئة بيانات ماينكرافت...`);
            return;
        }
        if (!this.bot || !this.bot.entity) {
            console.warn(`[BOT-${this.username}][THINK] كائن البوت أو الكيان غير متاح. تخطي دورة التفكير.`);
            return;
        }


        // الحصول على ملاحظة الحالة الحالية من بيئة Minecraft
        const currentState = this._getBotState();
        // تسجيل مؤشرات الحالة الرئيسية للتصحيح ومراقبة وضع البوت
        console.log(`[BOT-${this.username}][STATE] الحلقة: ${this.currentEpisode + 1} | الصحة:${currentState[0].toFixed(2)} الغذاء:${currentState[1].toFixed(2)} عدو:${currentState[4]} خشب:${currentState[8].toFixed(2)} حجر:${currentState[9].toFixed(2)} حديد:${currentState[13].toFixed(2)} فأس ألماس:${currentState[17]} بوابة نذر:${currentState[18]} في النذر:${currentState[19]} في النهاية:${currentState[20]} مات مؤخرًا:${currentState[21]} لاعب:${currentState[22]} مسافة لاعب:${currentState[23].toFixed(2)}`);

        let done = false; // علامة للإشارة إلى ما إذا كانت الحلقة الحالية قد انتهت

        // إذا لم تكن هذه هي التكة الأولى في الحلقة، قم بمعالجة التجربة السابقة
        if (this.lastState !== null && this.lastAction !== null) {
            // 1. حساب المكافأة المستلمة من الإجراء السابق وانتقال الحالة
            const reward = this._calculateReward(this.lastState, currentState, true); // بافتراض أن الإجراء كان ناجحًا
            this.currentEpisodeReward += reward; // تراكم إجمالي المكافأة للحلقة الحالية
            console.log(`[BOT-${this.username}][REWARD] مكافأة الإجراء السابق: ${reward.toFixed(2)}. إجمالي مكافأة الحلقة: ${this.currentEpisodeReward.toFixed(2)}`);

            // 2. تخزين tuple (state, action, reward, next_state, done) في ذاكرة إعادة التجربة
            this.qAgent.remember(this.lastState, this.lastAction, reward, currentState, done);
            // 3. تدريب نموذج Q-Network باستخدام دفعة عشوائية من ذاكرة إعادة التجربة
            await this.qAgent.trainModel();
        }

        // 4. اختيار الإجراء التالي بناءً على الحالة الحالية باستخدام سياسة وكيل Q-Agent
        const actionId = this.qAgent.chooseAction(currentState);
        console.log(`[BOT-${this.username}][DECISION] اختار الإجراء: ${ACTION_NAMES[actionId]} (إبسيلون: ${this.qAgent.epsilon.toFixed(4)})`);

        // منطق لمنع البوت من الوقوع في الخمول المفرط.
        // إذا كان خاملاً لعدد كبير جدًا من المرات المتتالية وكان الاستكشاف منخفضًا، فقم بزيادة الاستكشاف مؤقتًا.
        if (actionId === ACTIONS.IDLE) {
            this.consecutiveIdleActions++;
            if (this.consecutiveIdleActions > 15 && this.qAgent.epsilon < 0.1) {
                console.log(`[BOT-${this.username}][WARN] خمول مفرط، زيادة مؤقتة في الاستكشاف.`);
                this.qAgent.epsilon = 0.3; // زيادة إبسيلون مؤقتًا
                this.consecutiveIdleActions = 0; // إعادة تعيين العداد
            }
        } else {
            this.consecutiveIdleActions = 0; // إعادة تعيين العداد إذا لم يكن خاملاً
        }

        // 5. تنفيذ الإجراء المختار في بيئة Minecraft
        const { success, actionReward: immediateActionReward } = await this._executeBotAction(actionId);

        // تحديث `lastState` و `lastAction` لدورة التفكير التالية
        this.lastState = currentState;
        this.lastAction = actionId;

        // التحقق من شروط إنهاء الحلقة (مثل موت البوت)
        if (this.bot.health <= 0) {
            console.log(`[BOT-${this.username}][GAME_OVER] البوت مات! انتهت الحلقة.`);
            done = true; // وضع علامة على انتهاء الحلقة
            // إعطاء مكافأة سلبية كبيرة للموت والتدريب عليها فورًا
            this.qAgent.remember(this.lastState, this.lastAction, -100, currentState, true);
            await this.qAgent.trainModel(); // إجراء التدريب النهائي على تجربة الموت

            console.log(`[BOT-${this.username}][EPISODE_SUMMARY] انتهت الحلقة. المكافأة النهائية: ${this.currentEpisodeReward.toFixed(2)}`);
            this.currentEpisodeReward = 0; // إعادة تعيين مكافأة الحلقة للتشغيل التالي
            this.lastDiedTime = Date.now(); // تسجيل طابع زمني للموت لميزة "recentlyDied" في الحالة

            await this.qAgent.saveModel(); // حفظ أوزان النموذج المتعلمة

            // قطع اتصال البوت بأمان بعد الموت/انتهاء الحلقة.
            // سيتعامل BotManager مع إعادة التشغيل إذا لزم الأمر.
            this.disconnect();
        }
    }
}

// =============================================================================
// VII. فئة مدير البوتات (BotManager Class)
// =============================================================================
/**
 * يدير مثيلات متعددة من BotAgent، ويتعامل مع إنشائها، توصيلها، قطع اتصالها،
 * واستمراريتها (حفظ/تحميل ملفات تعريف البوتات ونماذجها).
 */
class BotManager {
    constructor() {
        this.bots = []; // مصفوفة لحفظ مثيلات BotAgent
    }

    /**
     * يحمل تكوينات البوتات الموجودة (أسماء المستخدمين) من ملف JSON.
     * يتيح ذلك للمدير إعادة توصيل نفس البوتات عند إعادة التشغيل.
     * @returns {Promise<Array<object>>} مصفوفة من كائنات تكوين البوت (مثال: [{ username: "Bot1" }]).
     * @private
     */
    async _loadBotConfigs() {
        try {
            // التأكد من وجود دليل النماذج قبل محاولة قراءة التكوينات
            await fs.mkdir(MODELS_DIR, { recursive: true });
            const data = await fs.readFile(BOTS_CONFIG_FILE, 'utf8');
            return JSON.parse(data);
        } catch (error) {
            // إذا لم يكن الملف موجودًا، فهذا سيناريو طبيعي للتشغيل الأول
            if (error.code === 'ENOENT') {
                console.log('[MANAGER][INFO] لم يتم العثور على ملف تكوينات بوتات موجود. بدء جديد.');
                return []; // إرجاع مصفوفة فارغة إذا لم يتم العثور على الملف
            }
            console.error('[MANAGER][ERROR] خطأ أثناء تحميل تكوينات البوتات:', error);
            return []; // إرجاع مصفوفة فارغة في حالة الأخطاء الأخرى أيضًا
        }
    }

    /**
     * يحفظ القائمة الحالية لتكوينات البوتات النشطة في ملف JSON.
     * هذا أمر حاسم للاستمرارية عبر عمليات إعادة تشغيل السكريبت.
     * @param {Array<object>} configs - مصفوفة من كائنات تكوين البوت المراد حفظها.
     * @private
     */
    async _saveBotConfigs(configs) {
        try {
            // التأكد من وجود دليل النماذج قبل حفظ التكوينات
            await fs.mkdir(MODELS_DIR, { recursive: true });
            await fs.writeFile(BOTS_CONFIG_FILE, JSON.stringify(configs, null, 2), 'utf8');
            console.log('[MANAGER][INFO] تم حفظ تكوينات البوتات بنجاح.');
        } catch (error) {
            console.error('[MANAGER][ERROR] خطأ أثناء حفظ تكوينات البوتات:', error);
        }
    }

    /**
     * يهيئ ويوصل البوتات بناءً على عدد محدد. يعطي الأولوية لإعادة توصيل
     * البوتات النشطة سابقًا من التكوينات المحفوظة.
     * @param {number} desiredBotCount - العدد الإجمالي للبوتات المراد تشغيلها.
     */
    async initializeBots(desiredBotCount) {
        const loadedConfigs = await this._loadBotConfigs();
        const existingUsernames = new Set(loadedConfigs.map(c => c.username));
        let activeBotConfigs = []; // لتخزين تكوينات البوتات التي ستكون نشطة في هذه الجلسة

        // 1. إعادة توصيل البوتات الموجودة من التكوينات المحملة، حتى الوصول إلى العدد المطلوب
        for (const config of loadedConfigs) {
            if (this.bots.length < desiredBotCount) {
                console.log(`[MANAGER][INFO] إعادة توصيل بوت موجود: ${config.username}`);
                const botAgent = new BotAgent(config.username);
                this.bots.push(botAgent);
                botAgent.connect();
                activeBotConfigs.push(config); // الاحتفاظ بهذا التكوين للحفظ الجديد
            } else {
                console.log(`[MANAGER][INFO] تجاهل البوت المحفوظ ${config.username} حيث تم الوصول إلى العدد المطلوب.`);
                // إذا كان العدد المطلوب أقل من العدد المحمل، قم بقطع اتصال البوتات الزائدة على الفور
                const tempBot = new BotAgent(config.username); // إنشاء وكيل مؤقت لحفظ نموذجه
                await tempBot.qAgent.saveModel(); // حفظ نموذجه حتى لو لم يكن يتصل
            }
        }

        // 2. إنشاء بوتات جديدة إذا لم يتم الوصول إلى العدد المطلوب بعد
        while (this.bots.length < desiredBotCount) {
            let newUsername = `RL_Agent_${Math.floor(Math.random() * 100000)}`;
            // التأكد من أن اسم المستخدم الجديد فريد
            while (existingUsernames.has(newUsername)) {
                newUsername = `RL_Agent_${Math.floor(Math.random() * 100000)}`;
            }
            existingUsernames.add(newUsername); // إضافته إلى المجموعة لمنع التكرار في هذه الجلسة

            console.log(`[MANAGER][INFO] إنشاء بوت جديد: ${newUsername}`);
            const botAgent = new BotAgent(newUsername);
            this.bots.push(botAgent);
            botAgent.connect();
            activeBotConfigs.push({ username: newUsername }); // إضافة تكوين البوت الجديد
        }

        // 3. حفظ المجموعة *الحالية* من تكوينات البوتات النشطة.
        // هذا يقوم بتحديث `bots_config.json` بالبوتات الفعلية التي تعمل في هذه الجلسة.
        await this._saveBotConfigs(activeBotConfigs);

        // اختياري: إعداد حفظ دوري لتكوينات البوتات (مثال: كل 5 دقائق)
        // هذا بمثابة حماية في حال تعطل السكريبت قبل إغلاقه بشكل نظيف.
        // setInterval(() => this._saveBotConfigs(this.bots.map(b => ({ username: b.username }))), 5 * 60 * 1000);
    }

    /**
     * يفصل جميع البوتات التي يتم إدارتها ويضمن حفظ نماذجها.
     * يتم استدعاء هذه الدالة عادة عند الإغلاق الآمن للسكريبت.
     */
    async disconnectAllBots() {
        console.log('[MANAGER][INFO] قطع اتصال جميع البوتات وحفظ النماذج...');
        const currentActiveConfigs = [];
        for (const botAgent of this.bots) {
            // قطع اتصال البوت إذا كان لا يزال متصلاً
            botAgent.disconnect();
            // التأكد من حفظ النموذج
            await botAgent.qAgent.saveModel();
            currentActiveConfigs.push({ username: botAgent.username });
        }
        // حفظ القائمة النهائية للبوتات النشطة للاستمرارية في التشغيل التالي
        await this._saveBotConfigs(currentActiveConfigs);
        console.log('[MANAGER][INFO] تم قطع اتصال جميع البوتات وحفظ النماذج.');
    }
}


// =============================================================================
// VIII. منطق التنفيذ الرئيسي (Main Execution Logic)
// =============================================================================
/**
 * نقطة الدخول الرئيسية للسكريبت. تدير عدد البوتات ودورة حياتها.
 * لم تعد بحاجة لإدخال المستخدم، حيث يتم تحديد عدد البوتات افتراضيًا في الكود.
 */
async function main() {
    // تحديد عدد البوتات المطلوب مباشرة من الثابت DEFAULT_BOT_COUNT
    const desiredBotCount = DEFAULT_BOT_COUNT;
    console.log(`[MANAGER][INFO] بدء تشغيل ${desiredBotCount} بوت (الإعداد الافتراضي).`);

    const botManager = new BotManager();
    await botManager.initializeBots(desiredBotCount);

    // --- معالجة الإغلاق الآمن (Graceful Shutdown Handling) ---
    // الاستماع لإشارات إنهاء العملية (Ctrl+C، أمر `kill`)
    process.on('SIGINT', async () => { // يتعامل مع Ctrl+C
        console.log('\n[PROCESS][INFO] تم اكتشاف SIGINT. إيقاف البوتات وحفظ النماذج...');
        await botManager.disconnectAllBots();
        process.exit(0); // الخروج بأمان
    });
    process.on('SIGTERM', async () => { // يتعامل مع أمر `kill`
        console.log('\n[PROCESS][INFO] تم اكتشاف SIGTERM. إيقاف البوتات وحفظ النماذج...');
        await botManager.disconnectAllBots();
        process.exit(0); // الخروج بأمان
    });

    // يمكنك إضافة معالجة أخطاء أكثر قوة للتعطلات غير المتوقعة هنا.
    // على سبيل المثال، استخدام try-catch حول الحلقة الرئيسية أو أداة PM2 لإدارة العمليات.
}

// بدء دالة التنفيذ الرئيسية
main();
