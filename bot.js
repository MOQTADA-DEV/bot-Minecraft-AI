// =============================================================================
// Minecraft Advanced RL Bot (Node.js)
// = =============================================================================
// This code implements an advanced Reinforcement Learning (RL) agent using
// TensorFlow.js for a Mineflayer bot in Minecraft Java Edition. It focuses on
// survival, resource gathering, progression towards advanced goals (Nether,
// Ender Dragon), and basic inter-bot communication. It also includes features
// for managing multiple bots and persisting their learned models across restarts.
//
// Please Note:
// - This script must be run in a Node.js environment, NOT in a web browser.
// - Full-scale implementations of algorithms like MuZero or DreamerV3 are beyond
//   the scope of a single Node.js script due to their immense computational
//   and infrastructural requirements. This code provides a robust DQN foundation
//   for complex in-game behaviors.
// - The `minecraft-data` library might not have specific data for very new
//   Minecraft patch versions (e.g., `1.21.5`). It will attempt to load the
//   closest available data, typically the major version (e.g., `1.21`).
//
// =============================================================================
// SETUP AND RUN INSTRUCTIONS:
// =============================================================================
// 1. Ensure Node.js is installed on your machine (https://nodejs.org/).
// 2. Create a new, empty folder for your project (e.g., `minecraft_rl_bots`).
// 3. Open your command-line interface (CMD, PowerShell, Git Bash, or Terminal)
//    and navigate to your new project folder.
// 4. Initialize a new Node.js project:
//    `npm init -y`
// 5. Install the necessary libraries:
//    `npm install mineflayer mineflayer-pathfinder @tensorflow/tfjs-node minecraft-data vec3`
// 6. Save this entire code block into a file named `main.js` (or `bot_manager.js`)
//    inside your project folder.
// 7. Make sure you have a Minecraft Java Edition server running at the specified
//    `SERVER_HOST` and `SERVER_PORT` below.
// 8. Run the bot manager script:
//    `node main.js`
//    The script will now start with a default number of bots, no prompt needed.
//
// =============================================================================


// =============================================================================
// I. LIBRARY IMPORTS
// =============================================================================
const mineflayer = require('mineflayer');
const pathfinder = require('mineflayer-pathfinder').pathfinder;
const goals = require('mineflayer-pathfinder').goals;
const { GoalBlock, GoalNear, GoalXZ } = require('mineflayer-pathfinder').goals;
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js for Node.js backend
const vec3 = require('vec3'); // Vector utilities for 3D coordinates
const fs = require('fs').promises; // Node.js File System module (Promise-based API)
// const readline = require('readline'); // No longer needed for user input


let mcData; // Minecraft game data (blocks, items, etc.) - initialized dynamically


// =============================================================================
// II. GLOBAL BOT CONFIGURATION & CONSTANTS
// =============================================================================
const SERVER_HOST = 'MQTADA12.aternos.me'; // Minecraft server address - UPDATED
const SERVER_PORT = 55865;             // Minecraft server port - UPDATED
const MINECRAFT_VERSION = '1.21.5';    // Minecraft version to connect to
const BOTS_CONFIG_FILE = './bots_config.json'; // File for saving/loading bot profiles
const MODELS_DIR = './bot_models';     // Directory where individual bot models are saved

// Default number of bots to run (can be changed directly in the code)
const DEFAULT_BOT_COUNT = 10; // UPDATED: Set a default number of bots


// =============================================================================
// III. DEFINING BOT ACTIONS
// =============================================================================
// This object maps descriptive action names to unique numeric IDs.
// Each ID corresponds to an output neuron in the Q-network.
const ACTIONS = {
    WALK_RANDOM: 0,
    ATTACK_NEAREST_HOSTILE: 1,
    FLEE_FROM_HOSTILE: 2,
    BUILD_BASIC_SHELTER: 3,
    DIG_WOOD: 4,
    CRAFT_WOODEN_PICKAXE: 5,
    COLLECT_DROPPED_ITEMS: 6,
    SLEEP: 7,
    IDLE: 8,
    MINE_STONE: 9,
    CRAFT_STONE_PICKAXE: 10,
    MINE_COAL: 11,
    MINE_IRON: 12,
    CRAFT_FURNACE: 13,
    SMELT_IRON: 14,
    CRAFT_IRON_PICKAXE: 15,
    MINE_DIAMONDS: 16,
    CRAFT_NETHER_PORTAL: 17,
    GO_TO_NETHER: 18,
    KILL_DRAGON: 19,
    COMMUNICATE_REQUEST_WOOD: 20,
    COMMUNICATE_OFFER_WOOD: 21,
    COMMUNICATE_ENEMY_WARNING: 22,
};
const ACTION_NAMES = Object.keys(ACTIONS); // Used for logging and readability


// =============================================================================
// IV. REINFORCEMENT LEARNING AGENT (Q-AGENT) CLASS
// =============================================================================
/**
 * Implements a Deep Q-Network (DQN) agent for Reinforcement Learning.
 * This agent uses TensorFlow.js to build and train a neural network
 * that approximates the Q-value function. Each bot will have its own QAgent.
 */
class QAgent {
    /**
     * @param {number} inputDim - Dimension of the state input vector.
     * @param {number} outputDim - Dimension of the action output vector (number of possible actions).
     * @param {string} modelId - A unique identifier for this agent's model (e.g., bot username).
     * @param {object} [params] - Optional hyperparameters for the agent.
     * @param {number} [params.learningRate=0.001] - Rate at which the model learns.
     * @param {number} [params.discountFactor=0.99] - Gamma, discounts future rewards.
     * @param {number} [params.epsilon=1.0] - Initial exploration rate (for epsilon-greedy policy).
     * @param {number} [params.epsilonDecay=0.9995] - Rate at which epsilon decays over time.
     * @param {number} [params.minEpsilon=0.005] - Minimum epsilon value.
     * @param {number} [params.replayBufferSize=200000] - Maximum size of the experience replay buffer.
     * @param {number} [params.batchSize=64] - Number of experiences to sample for each training step.
     * @param {number} [params.targetUpdateFreq=200] - Frequency (in training steps) to update the target network.
     */
    constructor(inputDim, outputDim, modelId, params = {}) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.modelId = modelId; // Used for unique model save path

        // Hyperparameters with defaults
        this.learningRate = params.learningRate !== undefined ? params.learningRate : 0.001;
        this.discountFactor = params.discountFactor !== undefined ? params.discountFactor : 0.99;
        this.epsilon = params.epsilon !== undefined ? params.epsilon : 1.0;
        this.epsilonDecay = params.epsilonDecay !== undefined ? params.epsilonDecay : 0.9995;
        this.minEpsilon = params.minEpsilon !== undefined ? params.minEpsilon : 0.005;

        this.replayBuffer = [];
        this.replayBufferSize = params.replayBufferSize !== undefined ? params.replayBufferSize : 200000;
        this.batchSize = params.batchSize !== undefined ? params.batchSize : 64;

        this.targetUpdateFreq = params.targetUpdateFreq !== undefined ? params.targetUpdateFreq : 200;
        this.updateCount = 0; // Tracks training steps for target network updates

        // Define unique path for this bot's model
        this.modelSavePath = `${MODELS_DIR}/${this.modelId}`;

        // Build main Q-network and target Q-network
        this.model = this._buildQNetwork();
        this.targetModel = this._buildQNetwork();
        this.targetModel.setWeights(this.model.getWeights()); // Synchronize weights initially

        // Attempt to load previously saved model weights for continuous learning
        this._loadModel();
    }

    /**
     * Builds the Deep Q-Network (DQN) model using TensorFlow.js sequential API.
     * This is a feedforward neural network. For advanced Recurrent RL (like LSTMs),
     * this method would be modified to include `tf.layers.lstm` or `tf.layers.gru`
     * layers, and the input shape would need to be adapted for sequences of states.
     * @returns {tf.LayersModel} The compiled TensorFlow.js model.
     * @private
     */
    _buildQNetwork() {
        const model = tf.sequential();
        // Input Layer: Receives the state vector
        model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [this.inputDim] }));
        // Hidden Layers: Process the input features
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        // Output Layer: Predicts Q-values for each possible action
        model.add(tf.layers.dense({ units: this.outputDim, activation: 'linear' })); // Linear activation for Q-values

        // Compile the model with an optimizer and a loss function
        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError' // Mean Squared Error is a common loss for Q-learning
        });
        return model;
    }

    /**
     * Attempts to load pre-trained model weights from the local file system.
     * This allows the bot to continue learning from previous sessions.
     * @private
     */
    async _loadModel() {
        try {
            // Check if the model directory exists for this specific bot
            await fs.access(this.modelSavePath);
            // Load model weights from the specified path
            this.model = await tf.loadLayersModel(`file://${this.modelSavePath}/model.json`);
            // Synchronize the target network with the loaded main model
            this.targetModel.setWeights(this.model.getWeights());
            console.log(`[RL_AGENT-${this.modelId}] Model loaded successfully from ${this.modelSavePath}.`);
            // When loading a trained model, it's common to start with a lower epsilon
            this.epsilon = this.minEpsilon;
        } catch (e) {
            // If the directory or model.json doesn't exist, it's a new model
            if (e.code === 'ENOENT') {
                console.log(`[RL_AGENT-${this.modelId}] No existing model found at ${this.modelSavePath}. Starting with a new model.`);
            } else {
                // Log other types of errors during model loading
                console.error(`[RL_AGENT-${this.modelId}] Error loading model from ${this.modelSavePath}:`, e);
            }
        }
    }

    /**
     * Saves the current model weights to the local file system.
     * This should be called periodically or when the bot disconnects,
     * ensuring learning progress is not lost.
     */
    async saveModel() {
        try {
            // Create the directory for this bot's model if it doesn't exist
            await fs.mkdir(this.modelSavePath, { recursive: true });
            // Save the main model to the specified path
            await this.model.save(`file://${this.modelSavePath}`);
            console.log(`[RL_AGENT-${this.modelId}] Model saved successfully to ${this.modelSavePath}.`);
        } catch (e) {
            console.error(`[RL_AGENT-${this.modelId}] Failed to save model to ${this.modelSavePath}:`, e);
        }
    }

    /**
     * Stores a single experience (state, action, reward, next state, done) in the
     * experience replay buffer. This buffer is crucial for batch training and
     * stabilizing the learning process in DQN.
     * @param {Array<number>} state - The current observed state of the environment.
     * @param {number} action - The action taken by the agent.
     * @param {number} reward - The immediate reward received from the environment.
     * @param {Array<number>} nextState - The state observed after taking the action.
     * @param {boolean} done - A boolean indicating if the episode has terminated.
     */
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push([state, action, reward, nextState, done]);
        // If the buffer exceeds its maximum size, remove the oldest experience
        if (this.replayBuffer.length > this.replayBufferSize) {
            this.replayBuffer.shift();
        }
    }

    /**
     * Selects an action based on the epsilon-greedy policy.
     * With probability `epsilon`, a random action is chosen (exploration).
     * Otherwise, the action with the highest predicted Q-value is chosen (exploitation).
     * @param {Array<number>} state - The current observed state.
     * @returns {number} The ID of the chosen action.
     */
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
            // Exploration: Choose a random action from all possible actions
            return Math.floor(Math.random() * this.outputDim);
        } else {
            // Exploitation: Predict Q-values for all actions in the current state
            const stateTensor = tf.tensor2d([state]);
            const qValues = this.model.predict(stateTensor);
            // Get the action ID with the maximum Q-value
            const action = qValues.argMax(-1).dataSync()[0];
            // Clean up TensorFlow tensors from memory to prevent memory leaks
            tf.dispose(stateTensor);
            tf.dispose(qValues);
            return action;
        }
    }

    /**
     * Trains the main Q-network using a mini-batch of experiences sampled
     * from the replay buffer. This implements the core of the DQN algorithm,
     * calculating target Q-values using the Bellman equation and updating the network.
     */
    async trainModel() {
        // Only train if there are enough experiences in the buffer to form a batch
        if (this.replayBuffer.length < this.batchSize) {
            return;
        }

        // Periodically update the target Q-network's weights to stabilize training
        this.updateCount++;
        if (this.updateCount % this.targetUpdateFreq === 0) {
            this.targetModel.setWeights(this.model.getWeights());
            console.log(`[RL_AGENT-${this.modelId}] Target network weights updated.`);
        }

        // Sample a random mini-batch of experiences from the replay buffer
        const batch = this.replayBuffer
            .sort(() => 0.5 - Math.random()) // Shuffle the buffer
            .slice(0, this.batchSize);       // Select the batch size from the shuffled experiences

        // Extract components of the experiences
        const states = batch.map(e => e[0]);
        const actions = batch.map(e => e[1]);
        const rewards = batch.map(e => e[2]);
        const nextStates = batch.map(e => e[3]);
        const dones = batch.map(e => e[4]);

        // Use tf.tidy to automatically dispose of intermediate tensors, preventing memory leaks
        await tf.tidy(async () => {
            const statesTensor = tf.tensor2d(states);
            const nextStatesTensor = tf.tensor2d(nextStates);

            // Predict Q-values for current states using the main network
            const currentQValues = this.model.predict(statesTensor);
            // Predict Q-values for next states using the target network (for more stable targets)
            const nextQValuesTarget = this.targetModel.predict(nextStatesTensor);

            // Convert current Q-values to a mutable JavaScript array for direct modification
            const targetQValues = currentQValues.arraySync();

            // Calculate the target Q-value for each experience in the batch
            for (let i = 0; i < this.batchSize; i++) {
                const isDone = dones[i];
                const reward = rewards[i];
                const action = actions[i];

                if (isDone) {
                    // If the episode terminated, the target Q-value for the taken action is just the immediate reward
                    targetQValues[i][action] = reward;
                } else {
                    // Otherwise, calculate using the Bellman equation: R + gamma * max(Q_target(s', a'))
                    const maxNextQ = tf.max(nextQValuesTarget.slice([i, 0], [1, this.outputDim])).dataSync()[0];
                    targetQValues[i][action] = reward + this.discountFactor * maxNextQ;
                }
            }

            // Convert the calculated target Q-values back to a TensorFlow tensor
            const targetsTensor = tf.tensor2d(targetQValues);

            // Train the main Q-network to predict these target Q-values.
            // This adjusts the network's weights to reduce the difference between predicted and target Q-values.
            await this.model.fit(statesTensor, targetsTensor, {
                epochs: 1, // Train for a single epoch per batch
                verbose: 0 // Suppress training output to keep console clean
            });
        });

        // Decay epsilon over time to reduce exploration and favor exploitation as agent learns
        this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    }
}


// =============================================================================
// V. BOT AGENT CLASS (Continuation from earlier thought block)
// =============================================================================
/**
 * Manages a single Mineflayer bot instance along with its dedicated Q-Agent.
 * Handles bot's lifecycle, state observation, action execution, and RL training.
 */
class BotAgent {
    /**
     * @param {string} username - The unique username for this bot.
     * @param {object} [agentParams={}] - Optional hyperparameters for the QAgent.
     */
    constructor(username, agentParams = {}) {
        this.username = username;
        this.bot = null; // Mineflayer bot instance
        this.qAgent = null; // QAgent instance for this bot

        // RL specific variables for the current episode
        this.lastState = null;
        this.lastAction = null;
        this.currentEpisodeReward = 0;
        this.consecutiveIdleActions = 0;
        this.lastDiedTime = 0; // Timestamp of last death for 'recentlyDied' state

        // Initialize QAgent for this bot
        // INPUT_NODES and OUTPUT_NODES are global constants for QAgent dimensions
        this.qAgent = new QAgent(INPUT_NODES, OUTPUT_NODES, this.username, agentParams);
    }

    /**
     * Connects the bot to the Minecraft server and sets up event listeners.
     */
    connect() {
        this.bot = mineflayer.createBot({
            host: SERVER_HOST,
            port: SERVER_PORT,
            username: this.username,
            version: MINECRAFT_VERSION
        });

        // Load the pathfinder plugin for intelligent movement for this bot
        this.bot.loadPlugin(pathfinder);

        // --- Register Bot Event Handlers ---
        this.bot.on('spawn', this._onSpawn.bind(this));      // When bot successfully connects and spawns
        this.bot.on('end', this._onEnd.bind(this));          // When bot disconnects
        this.bot.on('error', this._onError.bind(this));      // When an error occurs
        this.bot.on('chat', this._onChat.bind(this));        // When a chat message is received

        // Initialize mcData (Minecraft game data) globally if it hasn't been done yet.
        // This ensures mcData is available for all bots.
        if (!mcData) {
            this.bot.once('spawn', () => {
                mcData = require('minecraft-data')(this.bot.version);
                console.log(`[BOT-${this.username}][INIT] Minecraft data globally initialized for version: ${this.bot.version}`);
            });
        }
    }

    /**
     * Disconnects the bot from the server.
     */
    disconnect() {
        if (this.bot && this.bot.client.socket.readyState === 'open') {
            this.bot.quit();
        }
    }

    /**
     * Handles the bot's spawn event. Resets episode variables and starts the thinking loop.
     * @private
     */
    async _onSpawn() {
        console.log(`[BOT-${this.username}] Spawned into the world!`);
        this.bot.chat(`مرحباً! أنا ${this.username}, بوت ذكاء اصطناعي أتعلم البقاء والتطور.`); // Arabic greeting

        // Reset episode-specific variables for a new learning episode
        this.lastState = null;
        this.lastAction = null;
        this.currentEpisodeReward = 0;
        this.consecutiveIdleActions = 0;

        // Clear any existing thinking interval to prevent multiple loops
        if (this.thinkInterval) {
            clearInterval(this.thinkInterval);
        }
        // Start the bot's main thinking loop, calling _think() every 1 second
        this.thinkInterval = setInterval(this._think.bind(this), 1000);
        console.log(`[BOT-${this.username}] Thinking cycle started.`);
    }

    /**
     * Handles the bot's disconnection event. Clears the thinking loop and saves the model.
     * @param {string} reason - The reason for disconnection.
     * @private
     */
    async _onEnd(reason) {
        console.log(`[BOT-${this.username}] Disconnected: ${reason}`);
        if (this.thinkInterval) {
            clearInterval(this.thinkInterval); // Stop the thinking loop
        }
        await this.qAgent.saveModel(); // Save the learned model weights upon disconnection
        console.log(`[BOT-${this.username}] Model saved after disconnection.`);
    }

    /**
     * Handles unhandled errors that occur in the bot's operation. Saves the model.
     * @param {Error} err - The error object.
     * @private
     */
    async _onError(err) {
        console.error(`[BOT-${this.username}][ERROR] An unhandled error occurred: ${err.message}`);
        // Attempt to save the model on error to preserve learning progress
        await this.qAgent.saveModel();
    }

    /**
     * Handles incoming chat messages from other players or bots.
     * This is the basis for inter-bot communication and cooperation.
     * @param {string} username - The sender's username.
     * @param {string} message - The received message.
     * @private
     */
    async _onChat(username, message) {
        if (username === this.username) return; // Ignore messages sent by this bot itself

        console.log(`[BOT-${this.username}][CHAT_RX] From ${username}: "${message}"`);

        // --- Simple Message Parsing for Cooperative Actions ---
        // These are basic reactive responses. For advanced MARL, bots would learn
        // communication protocols and strategic coordination.

        if (message.includes('Need wood!')) {
            const woodCount = this._getBotState()[8] * 64; // Denormalize wood count for real value
            if (woodCount > 10) {
                this.bot.chat(`@${username} ${this.username}: لدي ${woodCount.toFixed(0)} خشب! يمكنني مشاركة البعض في موقعي.`); // Arabic response
            } else {
                this.bot.chat(`@${username} ${this.username}: ليس لدي الكثير من الخشب، لكنني سأحاول إيجاد بعض.`); // Arabic response
            }
        } else if (message.includes('Have extra wood!')) {
            this.bot.chat(`@${username} ${this.username}: شكراً للعرض! قد آتي لأجمع إذا احتجت ذلك.`); // Arabic response
        } else if (message.includes('Warning! Hostile mob')) {
            console.log(`[BOT-${this.username}][COOP_ACTION] Received enemy warning from ${username}. Prioritizing safety.`);
            const currentState = this._getBotState();
            // If health is low, strongly consider fleeing. Otherwise, prepare for attack.
            if (currentState[0] < 0.5) {
                await this._executeBotAction(ACTIONS.FLEE_FROM_HOSTILE);
            } else {
                // If not already aware of an enemy, attempt to attack
                if (currentState[4] === 0) {
                    await this._executeBotAction(ACTIONS.ATTACK_NEAREST_HOSTILE);
                }
            }
        }
    }

    /**
     * Gathers and normalizes the current state of the bot and its environment
     * into a numerical array suitable as input for the neural network.
     * This function defines the "observation space" of the agent,
     * including resources, health, enemy presence, and game progression.
     * @returns {number[]} An array of normalized numeric values representing the bot's state.
     * @private
     */
    _getBotState() {
        // Ensure mcData is available (should be initialized on first spawn)
        if (!mcData) {
            console.warn(`[BOT-${this.username}][STATE_ERR] mcData not initialized yet. Returning default state.`);
            return new Array(INPUT_NODES).fill(0); // Return an array of zeros if mcData is missing
        }

        // --- Basic Survival Stats (Normalized 0-1) ---
        const health = this.bot.health / 20;
        const food = this.bot.food / 20;
        const inWater = this.bot.entity.isInWater ? 1 : 0;
        const isDay = this.bot.time.timeOfDay < 13000 ? 1 : 0; // Approximate day (0-24000 ticks)

        // --- Enemy Presence (Normalized) ---
        const nearestHostile = this.bot.nearestEntity(entity =>
            entity.type === 'mob' && entity.mobType && entity.mobType.isHostile &&
            entity.position.distanceTo(this.bot.entity.position) < 30
        );
        const hasEnemyNearby = nearestHostile ? 1 : 0;
        const enemyDistance = nearestHostile ? nearestHostile.position.distanceTo(this.bot.entity.position) / 30 : 0; // 0-1 if within 30 blocks

        // --- Inventory & Resources (Normalized by max stack or common quantities) ---
        const hasFoodInInventory = this.bot.inventory.items().some(item => mcData.foodsArray.some(food => food.id === item.type)) ? 1 : 0;
        const woodCount = this.bot.inventory.count(mcData.itemsByName.oak_log.id) / 64; // Max stack 64
        const hasWoodenPickaxe = this.bot.inventory.items().some(item => item.name === 'wooden_pickaxe') ? 1 : 0;
        const stoneCount = this.bot.inventory.count(mcData.itemsByName.cobblestone.id) / 64;
        const hasStonePickaxe = this.bot.inventory.items().some(item => item.name === 'stone_pickaxe') ? 1 : 0;
        const coalCount = this.bot.inventory.count(mcData.itemsByName.coal.id) / 64;
        const ironOreCount = this.bot.inventory.count(mcData.itemsByName.iron_ore.id) / 64;
        const ironIngotCount = this.bot.inventory.count(mcData.itemsByName.iron_ingot.id) / 64;
        const hasIronPickaxe = this.bot.inventory.items().some(item => item.name === 'iron_pickaxe') ? 1 : 0;
        const obsidianCount = this.bot.inventory.count(mcData.itemsByName.obsidian.id) / 64;
        const hasDiamondPickaxe = this.bot.inventory.items().some(item => item.name === 'diamond_pickaxe') ? 1 : 0;
        const enderEyesCount = this.bot.inventory.count(mcData.itemsByName.ender_eye.id) / 12; // Need 12 for portal

        // --- Game Progression & Location (Binary flags) ---
        const netherPortalExistsNearby = !!this.bot.findBlock({
            matching: mcData.blocksByName.nether_portal.id,
            maxDistance: 100 // Check wider area for portals
        });
        const inNether = this.bot.entity.dimension === 'minecraft:the_nether' ? 1 : 0;
        const inEnd = this.bot.entity.dimension === 'minecraft:the_end' ? 1 : 0;

        // --- Basic "Memory" / Historical State (Conceptual for Recurrent RL) ---
        // A simple flag for recent events. A true RNN would process sequences of states.
        const recentlyDied = this.lastDiedTime && (Date.now() - this.lastDiedTime < 10000) ? 1 : 0; // Died in last 10 seconds

        // --- Multi-Agent Collaboration State ---
        const nearestOtherPlayer = this.bot.nearestEntity(entity => entity.type === 'player' && entity.username !== this.username);
        const hasOtherPlayerNearby = nearestOtherPlayer ? 1 : 0;
        const otherPlayerDistance = nearestOtherPlayer ? nearestOtherPlayer.position.distanceTo(this.bot.entity.position) / 60 : 0; // Normalize by 60 blocks

        return [
            health, food, inWater, isDay, hasEnemyNearby, enemyDistance,
            hasFoodInInventory, woodCount, hasWoodenPickaxe, stoneCount,
            hasStonePickaxe, coalCount, ironOreCount, ironIngotCount,
            hasIronPickaxe, obsidianCount, hasDiamondPickaxe, enderEyesCount,
            netherPortalExistsNearby, inNether, inEnd, // These are already 0/1 from !! conversion
            recentlyDied, hasOtherPlayerNearby, otherPlayerDistance
        ];
    }

    /**
     * Executes a specific bot action based on its ID. This function contains the
     * complex logic for multi-step actions and interacts with the Mineflayer API.
     * @param {number} actionId - The ID of the action to execute.
     * @returns {Promise<{success: boolean, actionReward: number}>} Object indicating success and immediate reward.
     * @private
     */
    async _executeBotAction(actionId) {
        console.log(`[BOT-${this.username}][ACTION_EXEC] Attempting to execute: ${ACTION_NAMES[actionId]}`);
        let success = false;
        let actionReward = 0; // Immediate reward for this specific action

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
                    } catch (err) {
                        console.log(`[BOT-${this.username}][ACTION_ERR] Random walk to ${randomX.toFixed(0)}, ${randomZ.toFixed(0)} failed: ${err.message}`);
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
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No hostile mob nearby to attack.`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Flee attempt failed: ${err.message}`);
                            actionReward = -0.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No hostile mob to flee from.`);
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
                                console.log(`[BOT-${this.username}][ACTION_INFO] No dirt item in inventory despite count.`);
                                actionReward = -0.2;
                                break;
                            }
                            await this.bot.setQuickBarSlot(dirtItem.slot);
                            const currentPos = this.bot.entity.position.floor();

                            // Build a 3x3 base (floor)
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
                            // Build 3 layers of walls (simple box)
                            for (let y = 0; y < 3; y++) {
                                for (let x = -1; x <= 1; x++) {
                                    for (let z = -1; z <= 1; z++) {
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
                            console.log(`[BOT-${this.username}][ACTION_INFO] Basic dirt shelter constructed.`);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Building failed: ${err.message}`);
                            actionReward = -1.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Not enough dirt for building (need 16).`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Digging wood failed: ${err.message}`);
                            actionReward = -0.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No wood block found nearby to dig.`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Crafting wooden pickaxe failed: ${err.message}`);
                            actionReward = -1.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No wooden pickaxe recipe available or missing materials.`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Collecting item failed: ${err.message}`);
                            actionReward = -0.3;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No dropped items nearby to collect.`);
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
                    if (bed && !this._getBotState()[3]) {
                        try {
                            await this.bot.sleep(bed);
                            actionReward = 1.0;
                            success = true;
                            setTimeout(async () => { await this.bot.wake(); }, 5000);
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Sleeping failed: ${err.message}`);
                            actionReward = -0.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No bed found or not night time.`);
                        actionReward = -0.2;
                    }
                    break;
                }

                case ACTIONS.IDLE: {
                    actionReward = 0.05;
                    success = true;
                    break;
                }

                case ACTIONS.MINE_STONE: {
                    const stoneBlock = this.bot.findBlock({
                        matching: mcData.blocksByName.stone.id,
                        maxDistance: 16
                    });
                    const hasWoodenPick = this._getBotState()[9];
                    if (stoneBlock && this.bot.canDigBlock(stoneBlock) && hasWoodenPick) {
                        try {
                            await this.bot.dig(stoneBlock);
                            actionReward = 1.8;
                            success = true;
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Mining stone failed: ${err.message}`);
                            actionReward = -0.6;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No stone found or missing wooden pickaxe.`);
                        actionReward = -0.3;
                    }
                    break;
                }

                case ACTIONS.CRAFT_STONE_PICKAXE: {
                    const stonePickaxeRecipe = this.bot.recipesFor(mcData.itemsByName.stone_pickaxe.id).find(r => true);
                    const hasEnoughStone = this._getBotState()[10] * 64 >= 3;
                    if (stonePickaxeRecipe && hasEnoughStone) {
                        try {
                            await this.bot.craft(stonePickaxeRecipe, 1);
                            actionReward = 3.0;
                            success = true;
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Crafting stone pickaxe failed: ${err.message}`);
                            actionReward = -1.2;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No stone pickaxe recipe or missing materials (3 cobblestone).`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Mining coal failed: ${err.message}`);
                            actionReward = -0.8;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No coal ore found or missing suitable pickaxe.`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Mining iron failed: ${err.message}`);
                            actionReward = -1.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No iron ore found or missing suitable pickaxe.`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Crafting furnace failed: ${err.message}`);
                            actionReward = -1.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No furnace recipe or missing materials (8 cobblestone).`);
                        actionReward = -0.7;
                    }
                    break;
                }

                case ACTIONS.SMELT_IRON: {
                    const hasFurnaceInInventory = this.bot.inventory.items().some(item => item.name === 'furnace');
                    const hasCoal = this._getBotState()[12] * 64 >= 1;
                    const hasRawIron = this._getBotState()[13] * 64 >= 1;

                    if (hasFurnaceInInventory && hasCoal && hasRawIron) {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Attempting to smelt iron (conceptual).`);
                        actionReward = 5.0;
                        success = true;
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Cannot smelt: Missing furnace, coal, or raw iron.`);
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
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Crafting iron pickaxe failed: ${err.message}`);
                            actionReward = -2.5;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No iron pickaxe recipe or missing materials (3 iron ingots).`);
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
                            console.log(`[BOT-${this.username}][ACTION_INFO] Found diamond ore, attempting to mine.`);
                            try {
                                await this.bot.dig(diamondOre);
                                actionReward = 15.0;
                                success = true;
                            } catch (err) {
                                console.log(`[BOT-${this.username}][ACTION_ERR] Mining diamonds failed: ${err.message}`);
                                actionReward = -3.0;
                            }
                        } else {
                            console.log(`[BOT-${this.username}][ACTION_INFO] No diamond ore found nearby.`);
                            actionReward = -1.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No suitable pickaxe (iron or diamond) to mine diamonds.`);
                        actionReward = -1.5;
                    }
                    break;
                }

                case ACTIONS.CRAFT_NETHER_PORTAL: {
                    const obsidianCount = this._getBotState()[16] * 64;
                    const hasFlintAndSteel = this.bot.inventory.items().some(item => item.name === 'flint_and_steel');

                    if (obsidianCount >= 14 && hasFlintAndSteel) {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Attempting to craft a Nether Portal (conceptual).`);
                        actionReward = 50.0;
                        success = true;
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Not enough obsidian (need 14) or no flint and steel to craft portal.`);
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
                        console.log(`[BOT-${this.username}][ACTION_INFO] Found Nether Portal, attempting to enter.`);
                        try {
                            await this.bot.pathfinder.goto(new GoalBlock(netherPortalBlock.position.x, netherPortalBlock.position.y, netherPortalBlock.position.z));
                            actionReward = 100.0;
                            success = true;
                        } catch (err) {
                            console.log(`[BOT-${this.username}][ACTION_ERR] Failed to go to Nether Portal: ${err.message}`);
                            actionReward = -10.0;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No Nether Portal found nearby to enter.`);
                        actionReward = -2.0;
                    }
                    break;
                }

                case ACTIONS.KILL_DRAGON: {
                    console.log(`[BOT-${this.username}][ACTION_INFO] Attempting to kill the Ender Dragon (conceptual action).`);
                    if (this._getBotState()[20] === 1) {
                        actionReward = 1000.0;
                        success = true;
                        console.log(`[BOT-${this.username}][ACTION_INFO] Placeholder: Bot would now execute complex Ender Dragon fight logic.`);
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Not in The End dimension to kill dragon.`);
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
                            this.bot.chat(`@${nearestPlayer.username} ${this.username}: I need wood! Can you help?`);
                            actionReward = 0.2;
                            success = true;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Not needing wood or no other players nearby to request from.`);
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
                            this.bot.chat(`@${nearestPlayer.username} ${this.username}: I have extra wood if you need some!`);
                            actionReward = 0.2;
                            success = true;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] Not enough wood to offer or no other players nearby.`);
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
                            this.bot.chat(`@${nearestPlayer.username} ${this.username}: Warning! Hostile mob (${enemy.displayName}) at my location: ${enemy.position.x.toFixed(0)}, ${enemy.position.y.toFixed(0)}, ${enemy.position.z.toFixed(0)}!`);
                            actionReward = 0.5;
                            success = true;
                        }
                    } else {
                        console.log(`[BOT-${this.username}][ACTION_INFO] No enemy to warn about or no other players nearby.`);
                        actionReward = -0.1;
                    }
                    break;
                }

                default: {
                    console.log(`[BOT-${this.username}][ACTION_ERR] Unknown action ID: ${actionId}`);
                    actionReward = -0.05;
                    break;
                }
            }
        } catch (e) {
            console.error(`[BOT-${this.username}][ACTION_EXCEPTION] Exception during action ${ACTION_NAMES[actionId]}: ${e.message}`);
            actionReward = -0.7;
        }

        return { success, actionReward };
    }


    /**
     * Calculates the reward received by the bot based on the transition from an
     * old state to a new state after taking an action. This is crucial for
     * guiding the RL agent's learning process.
     * Reward shaping is used to encourage desired behaviors and discourage undesirable ones.
     * @param {Array<number>} oldState - The state of the bot before the action.
     * @param {Array<number>} newState - The state of the bot after the action.
     * @param {boolean} actionSuccess - Indicates if the executed action was successful.
     * @returns {number} The calculated reward value.
     * @private
     */
    _calculateReward(oldState, newState, actionSuccess) {
        let reward = 0;

        // --- Base Survival & Health ---
        reward += 0.01; // Small positive reward for simply staying alive each tick
        reward += (newState[0] - oldState[0]) * 10; // Significant reward for health gain, large penalty for health loss
        reward += (newState[1] - oldState[1]) * 5;  // Reward for food gain, penalty for loss
        if (newState[0] < 0.2 && newState[0] < oldState[0]) { reward -= 15; } // Severe penalty for critically low health
        if (newState[1] < 0.2 && newState[1] < oldState[1]) { reward -= 10; } // Severe penalty for critically low food
        if (newState[21] === 1 && oldState[21] === 0) { reward -= 50; } // Large penalty for dying (recentlyDied flag)

        // --- Enemy Interaction ---
        if (newState[4] === 0 && oldState[4] === 1) { reward += 20; } // Big reward for successfully escaping/eliminating hostile
        else if (newState[4] === 1 && oldState[4] === 1 && newState[5] < oldState[5]) { reward -= 3; } // Penalty if enemy gets closer
        else if (newState[4] === 1 && oldState[4] === 0) { reward -= 7; } // Penalty for new enemy appearing nearby

        // --- Resource Gathering Progression (Tier-based rewards) ---
        // Wood
        if (newState[8] > oldState[8]) { reward += (newState[8] - oldState[8]) * 5; }
        // Stone/Cobblestone
        if (newState[9] > oldState[9]) { reward += (newState[9] - oldState[9]) * 7; }
        // Coal
        if (newState[11] > oldState[11]) { reward += (newState[11] - oldState[11]) * 8; }
        // Iron Ore
        if (newState[12] > oldState[12]) { reward += (newState[12] - oldState[12]) * 15; }
        // Iron Ingots (from smelting)
        if (newState[13] > oldState[13]) { reward += (newState[13] - oldState[13]) * 20; }
        // Obsidian
        if (newState[16] > oldState[16]) { reward += (newState[16] - oldState[16]) * 30; }
        // Ender Eyes
        if (newState[18] > oldState[18]) { reward += (newState[18] - oldState[18]) * 40; }

        // --- Tool/Crafting Progression (Milestone rewards) ---
        if (newState[9] === 1 && oldState[9] === 0) { reward += 10; } // Crafted Wooden Pickaxe
        if (newState[11] === 1 && oldState[11] === 0) { reward += 20; } // Crafted Stone Pickaxe
        if (newState[15] === 1 && oldState[15] === 0) { reward += 40; } // Crafted Iron Pickaxe
        if (newState[17] === 1 && oldState[17] === 0) { reward += 80; } // Crafted Diamond Pickaxe

        // --- Major Game Progression Goals (Very High Rewards) ---
        if (newState[18] === 1 && oldState[18] === 0) { reward += 200; } // Nether Portal Built
        if (newState[19] === 1 && oldState[19] === 0) { reward += 500; } // Entered Nether
        if (newState[20] === 1 && oldState[20] === 0) { reward += 1000; } // Entered The End
        // Hypothetical HUGE reward for killing Ender Dragon (would be triggered by game event)
        // if (DRAGON_KILLED_GLOBAL_EVENT) { reward += 5000; }

        // --- Multi-Agent Collaboration Rewards ---
        if (newState[23] === 1 && oldState[23] === 0) { reward += 0.5; }

        // --- Penalty for Failed Actions ---
        if (!actionSuccess) {
            reward -= 1.0;
        }

        return reward;
    }

    /**
     * The main thinking loop for this specific bot agent.
     * Perceives the environment, makes decisions, executes actions, and learns.
     * This function is called repeatedly by `setInterval`.
     * @private
     */
    async _think() {
        // Ensure Minecraft data is loaded before proceeding with state observation
        if (!mcData) {
            console.log(`[BOT-${this.username}][THINK] Waiting for Minecraft data to initialize...`);
            return;
        }

        // Get the current state observation from the Minecraft environment
        const currentState = this._getBotState();
        // Log key state indicators for debugging and monitoring bot's situation
        console.log(`[BOT-${this.username}][STATE] H:${currentState[0].toFixed(2)} F:${currentState[1].toFixed(2)} E:${currentState[4]} W:${currentState[8].toFixed(2)} S:${currentState[9].toFixed(2)} I:${currentState[13].toFixed(2)} DP:${currentState[17]} NP:${currentState[18]} N:${currentState[19]} E:${currentState[20]} Died:${currentState[21]} Player:${currentState[22]} Dist:${currentState[23].toFixed(2)}`);

        let done = false; // Flag to indicate if the current episode has ended

        // If this is not the first tick in an episode, process the previous experience
        if (this.lastState !== null && this.lastAction !== null) {
            // 1. Calculate the reward received from the previous action and state transition
            const reward = this._calculateReward(this.lastState, currentState, true); // Assuming action was successful
            this.currentEpisodeReward += reward; // Accumulate total reward for the current episode
            console.log(`[BOT-${this.username}][REWARD] Prev action reward: ${reward.toFixed(2)}. Total Episode Reward: ${this.currentEpisodeReward.toFixed(2)}`);

            // 2. Store the (state, action, reward, next_state, done) tuple in the replay buffer
            this.qAgent.remember(this.lastState, this.lastAction, reward, currentState, done);
            // 3. Train the Q-Network model using a random batch from the replay buffer
            await this.qAgent.trainModel();
        }

        // 4. Choose the next action based on the current state using the Q-agent's policy
        const actionId = this.qAgent.chooseAction(currentState);
        console.log(`[BOT-${this.username}][DECISION] Chose action: ${ACTION_NAMES[actionId]} (Epsilon: ${this.qAgent.epsilon.toFixed(4)})`);

        // Logic to prevent the bot from getting stuck idling excessively.
        // If it idles too many times consecutively and exploration is low, temporarily boost exploration.
        if (actionId === ACTIONS.IDLE) {
            this.consecutiveIdleActions++;
            if (this.consecutiveIdleActions > 15 && this.qAgent.epsilon < 0.1) {
                console.log(`[BOT-${this.username}][INFO] Idling excessively, temporarily boosting exploration.`);
                this.qAgent.epsilon = 0.3; // Temporarily increase epsilon
                this.consecutiveIdleActions = 0; // Reset counter
            }
        } else {
            this.consecutiveIdleActions = 0; // Reset counter if not idling
        }

        // 5. Execute the chosen action in the Minecraft environment
        const { success, actionReward: immediateActionReward } = await this._executeBotAction(actionId);

        // Update `lastState` and `lastAction` for the next thinking cycle
        this.lastState = currentState;
        this.lastAction = actionId;

        // Check for episode termination conditions (e.g., bot dies)
        if (this.bot.health <= 0) {
            console.log(`[BOT-${this.username}][GAME_OVER] Bot died! Episode terminated.`);
            done = true; // Mark episode as done
            // Give a significant negative reward for death and train on it immediately
            this.qAgent.remember(this.lastState, this.lastAction, -100, currentState, true);
            await this.qAgent.trainModel(); // Perform final training on death experience

            console.log(`[BOT-${this.username}][EPISODE_SUMMARY] Episode ended. Final reward: ${this.currentEpisodeReward.toFixed(2)}`);
            this.currentEpisodeReward = 0; // Reset episode reward for the next run
            this.lastDiedTime = Date.now(); // Record death timestamp for 'recentlyDied' state

            await this.qAgent.saveModel(); // Save the learned model weights

            // Disconnect the bot gracefully after death/episode end.
            // The BotManager will handle restarting if needed.
            this.disconnect();
        }
    }
}

// =============================================================================
// XI. BOT MANAGER CLASS
// =============================================================================
/**
 * Manages multiple BotAgent instances, handling their creation, connection,
 * disconnection, and persistence (saving/loading bot profiles and models).
 */
class BotManager {
    constructor() {
        this.bots = []; // Array to hold BotAgent instances
    }

    /**
     * Loads existing bot configurations (usernames) from a JSON file.
     * This allows the manager to reconnect the same bots on restart.
     * @returns {Promise<Array<object>>} An array of bot configuration objects (e.g., [{ username: "Bot1" }]).
     * @private
     */
    async _loadBotConfigs() {
        try {
            // Ensure the models directory exists before attempting to read configs
            await fs.mkdir(MODELS_DIR, { recursive: true });
            const data = await fs.readFile(BOTS_CONFIG_FILE, 'utf8');
            return JSON.parse(data);
        } catch (error) {
            // If the file doesn't exist, it's a normal scenario for the first run
            if (error.code === 'ENOENT') {
                console.log('[MANAGER] No existing bot configurations file found. Starting fresh.');
                return []; // Return empty array if file not found
            }
            console.error('[MANAGER] Error loading bot configurations:', error);
            return []; // Return empty array on other errors too
        }
    }

    /**
     * Saves the current list of active bot configurations to a JSON file.
     * This is crucial for persistence across script restarts.
     * @param {Array<object>} configs - An array of bot configuration objects to save.
     * @private
     */
    async _saveBotConfigs(configs) {
        try {
            // Ensure the models directory exists before saving configs
            await fs.mkdir(MODELS_DIR, { recursive: true });
            await fs.writeFile(BOTS_CONFIG_FILE, JSON.stringify(configs, null, 2), 'utf8');
            console.log('[MANAGER] Bot configurations saved successfully.');
        } catch (error) {
            console.error('[MANAGER] Error saving bot configurations:', error);
        }
    }

    /**
     * Initializes and connects bots based on a specified count. It prioritizes
     * reconnecting previously active bots from saved configurations.
     * @param {number} desiredBotCount - The total number of bots to ensure are running.
     */
    async initializeBots(desiredBotCount) {
        const loadedConfigs = await this._loadBotConfigs();
        const existingUsernames = new Set(loadedConfigs.map(c => c.username));
        let activeBotConfigs = []; // To store configurations of bots that will be active in this session

        // 1. Reconnect existing bots from loaded configurations, up to desired count
        for (const config of loadedConfigs) {
            if (this.bots.length < desiredBotCount) {
                console.log(`[MANAGER] Reconnecting existing bot: ${config.username}`);
                const botAgent = new BotAgent(config.username);
                this.bots.push(botAgent);
                botAgent.connect();
                activeBotConfigs.push(config); // Keep this config for the new save
            } else {
                console.log(`[MANAGER] Discarding saved bot ${config.username} as desired count is met.`);
                // If the desired count is lower than loaded, disconnect the excess bots immediately
                const tempBot = new BotAgent(config.username); // Create a temporary agent to save its model
                await tempBot.qAgent.saveModel(); // Save its model even if not connecting
            }
        }

        // 2. Create new bots if the desired count is not yet met
        while (this.bots.length < desiredBotCount) {
            let newUsername = `RL_Agent_${Math.floor(Math.random() * 100000)}`;
            // Ensure the new username is unique
            while (existingUsernames.has(newUsername)) {
                newUsername = `RL_Agent_${Math.floor(Math.random() * 100000)}`;
            }
            existingUsernames.add(newUsername); // Add to set to prevent duplicates in this session

            console.log(`[MANAGER] Creating new bot: ${newUsername}`);
            const botAgent = new BotAgent(newUsername);
            this.bots.push(botAgent);
            botAgent.connect();
            activeBotConfigs.push({ username: newUsername }); // Add new bot's config
        }

        // 3. Save the *current* set of active bot configurations.
        // This updates `bots_config.json` with the actual bots running in this session.
        await this._saveBotConfigs(activeBotConfigs);

        // Optional: Set up a periodic save for bot configurations (e.g., every 5 minutes)
        // This is a failsafe in case the script crashes before a clean shutdown.
        // setInterval(() => this._saveBotConfigs(this.bots.map(b => ({ username: b.username }))), 5 * 60 * 1000);
    }

    /**
     * Disconnects all managed bots and ensures their models are saved.
     * This function is typically called on graceful shutdown.
     */
    async disconnectAllBots() {
        console.log('[MANAGER] Disconnecting all bots and saving models...');
        const currentActiveConfigs = [];
        for (const botAgent of this.bots) {
            // Disconnect bot if it's still connected
            botAgent.disconnect();
            // Ensure the model is saved
            await botAgent.qAgent.saveModel();
            currentActiveConfigs.push({ username: botAgent.username });
        }
        // Save the final list of active bots to persist for the next run
        await this._saveBotConfigs(currentActiveConfigs);
        console.log('[MANAGER] All bots disconnected and models saved.');
    }
}


// =============================================================================
// XII. MAIN EXECUTION LOGIC
// =============================================================================
/**
 * The main entry point of the script. Handles user input for the number of bots
 * and manages their lifecycle.
 */
async function main() {
    // No longer need readline as desiredBotCount is set directly
    // const rl = readline.createInterface({
    //     input: process.stdin,
    //     output: process.stdout
    // });

    // Set desiredBotCount directly from the constant
    const desiredBotCount = DEFAULT_BOT_COUNT;
    console.log(`[MANAGER] Starting with ${desiredBotCount} bots (default setting).`);

    const botManager = new BotManager();
    await botManager.initializeBots(desiredBotCount);

    // --- Graceful Shutdown Handling ---
    // Listen for process termination signals (Ctrl+C, `kill` command)
    process.on('SIGINT', async () => { // Handles Ctrl+C
        console.log('\n[PROCESS] Detected SIGINT. Shutting down bots and saving models...');
        await botManager.disconnectAllBots();
        process.exit(0); // Exit gracefully
    });
    process.on('SIGTERM', async () => { // Handles `kill` command
        console.log('\n[PROCESS] Detected SIGTERM. Shutting down bots and saving models...');
        await botManager.disconnectAllBots();
        process.exit(0); // Exit gracefully
    });

    // You might want to add more robust error handling for unexpected crashes
    // For example, a try-catch around the main loop or PM2 for process management
}

// Start the main execution function
main();
