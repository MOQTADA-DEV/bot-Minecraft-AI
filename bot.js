// =============================================================================
// Minecraft Advanced RL Bot (Node.js)
// = =============================================================================
// This code implements an advanced Reinforcement Learning (RL) agent using
// TensorFlow.js for a Mineflayer bot in Minecraft Java Edition.
// It focuses on survival, resource gathering, progression towards advanced goals
// (Nether, Ender Dragon), and basic inter-bot communication.
//
// Please Note:
// - This script must be run in a Node.js environment, NOT in a web browser.
// - Full-scale implementations of algorithms like MuZero or DreamerV3 are beyond
//   the scope of a single Node.js script due to their immense computational
//   and infrastructural requirements. This code provides a robust Q-learning
//   foundation for complex in-game behaviors.
//
// =============================================================================
// SETUP AND RUN INSTRUCTIONS:
// =============================================================================
// 1. Ensure Node.js is installed on your machine (https://nodejs.org/).
// 2. Create a new, empty folder for your project (e.g., `minecraft_rl_bot`).
// 3. Open your command-line interface (CMD, PowerShell, Git Bash, or Terminal)
//    and navigate to your new project folder.
// 4. Initialize a new Node.js project:
//    `npm init -y`
// 5. Install the necessary libraries:
//    `npm install mineflayer mineflayer-pathfinder @tensorflow/tfjs-node minecraft-data vec3`
// 6. Save this entire code block into a file named `bot.js` inside your project folder.
// 7. Make sure you have a Minecraft Java Edition server running (e.g., a local
//    server or a dedicated server) at the specified `host` and `port` below.
// 8. Run the bot script:
//    `node bot.js`
//
// =============================================================================


// =============================================================================
// I. LIBRARY IMPORTS
// =============================================================================
const mineflayer = require('mineflayer');
const pathfinder = require('mineflayer-pathfinder').pathfinder;
const goals = require('mineflayer-pathfinder').goals;
const { GoalBlock, GoalNear, GoalXZ } = require('mineflayer-pathfinder').goals;
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js for Node.js
const vec3 = require('vec3'); // Vector utilities for 3D coordinates

let mcData; // Minecraft game data (blocks, items, etc.) - initialized after bot connects


// =============================================================================
// II. BOT CONFIGURATION
// =============================================================================
const BOT_CONFIG = {
    host: 'localhost',       // Minecraft server address (e.g., 'localhost' or '127.0.0.1' for local)
    port: 25565,             // Minecraft server port (default is 25565)
    username: 'RL_Agent_' + Math.floor(Math.random() * 10000), // Unique name for each bot instance
    version: '1.16.5'        // Minecraft version to connect to (must match server)
};

// Create the Mineflayer bot instance
const bot = mineflayer.createBot(BOT_CONFIG);

// Load the pathfinder plugin for intelligent movement
bot.loadPlugin(pathfinder);

// Initialize Minecraft data specific to the connected version once the bot spawns
bot.once('spawn', () => {
    mcData = require('minecraft-data')(bot.version);
    console.log(`[INIT] Minecraft data initialized for version: ${bot.version}`);
});


// =============================================================================
// III. DEFINING BOT ACTIONS
// =============================================================================
// This object maps descriptive action names to unique numeric IDs.
// Each ID corresponds to an output neuron in the Q-network.
const ACTIONS = {
    WALK_RANDOM: 0,                   // Move to a random nearby location
    ATTACK_NEAREST_HOSTILE: 1,        // Attack the closest hostile mob
    FLEE_FROM_HOSTILE: 2,             // Run away from the closest hostile mob
    BUILD_BASIC_SHELTER: 3,           // Construct a simple dirt shelter
    DIG_WOOD: 4,                      // Mine the nearest wood log
    CRAFT_WOODEN_PICKAXE: 5,          // Craft a wooden pickaxe
    COLLECT_DROPPED_ITEMS: 6,         // Pick up nearby dropped items
    SLEEP: 7,                         // Attempt to sleep in a bed (if night & bed nearby)
    IDLE: 8,                          // Do nothing for a short period
    MINE_STONE: 9,                    // Mine cobblestone
    CRAFT_STONE_PICKAXE: 10,          // Craft a stone pickaxe
    MINE_COAL: 11,                    // Mine coal ore
    MINE_IRON: 12,                    // Mine iron ore
    CRAFT_FURNACE: 13,                // Craft a furnace
    SMELT_IRON: 14,                   // Smelt iron ore into ingots
    CRAFT_IRON_PICKAXE: 15,           // Craft an iron pickaxe
    MINE_DIAMONDS: 16,                // Attempt to mine diamond ore (requires iron/diamond pick)
    CRAFT_NETHER_PORTAL: 17,          // Attempt to craft a Nether Portal (conceptual)
    GO_TO_NETHER: 18,                 // Attempt to enter the Nether dimension (conceptual)
    KILL_DRAGON: 19,                  // Attempt to kill the Ender Dragon (highly conceptual)
    COMMUNICATE_REQUEST_WOOD: 20,     // Chat: Request wood from other players/bots
    COMMUNICATE_OFFER_WOOD: 21,       // Chat: Offer wood to other players/bots
    COMMUNICATE_ENEMY_WARNING: 22,    // Chat: Warn about nearby hostile mobs
};
// Array of action names for easier logging and readability
const ACTION_NAMES = Object.keys(ACTIONS);


// =============================================================================
// IV. REINFORCEMENT LEARNING AGENT (Q-AGENT)
// =============================================================================
/**
 * Implements a Deep Q-Network (DQN) agent for Reinforcement Learning.
 * This agent uses TensorFlow.js to build and train a neural network
 * that approximates the Q-value function.
 */
class QAgent {
    /**
     * @param {number} inputDim - Dimension of the state input vector.
     * @param {number} outputDim - Dimension of the action output vector (number of possible actions).
     * @param {object} [params] - Optional parameters for the agent.
     * @param {number} [params.learningRate=0.001] - Rate at which the model learns.
     * @param {number} [params.discountFactor=0.99] - Gamma, discounts future rewards.
     * @param {number} [params.epsilon=1.0] - Initial exploration rate (for epsilon-greedy policy).
     * @param {number} [params.epsilonDecay=0.9995] - Rate at which epsilon decays over time.
     * @param {number} [params.minEpsilon=0.005] - Minimum epsilon value.
     * @param {number} [params.replayBufferSize=200000] - Maximum size of the experience replay buffer.
     * @param {number} [params.batchSize=64] - Number of experiences to sample for each training step.
     * @param {number} [params.targetUpdateFreq=200] - Frequency (in training steps) to update the target network.
     */
    constructor(inputDim, outputDim, params = {}) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;

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
        this.updateCount = 0; // Tracks training steps to update target network

        // Build main Q-network and target Q-network
        this.model = this._buildQNetwork();
        this.targetModel = this._buildQNetwork();
        this.targetModel.setWeights(this.model.getWeights()); // Synchronize weights initially

        // Attempt to load previously saved model weights for continuous learning
        this._loadModel();
    }

    /**
     * Builds the Deep Q-Network (DQN) model using TensorFlow.js sequential API.
     * This is a feedforward neural network. For Recurrent RL, this would
     * involve adding `tf.layers.lstm` or `tf.layers.gru` layers and
     * adapting input shape for sequences.
     * @returns {tf.LayersModel} The compiled TensorFlow.js model.
     * @private
     */
    _buildQNetwork() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [this.inputDim] }));
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.outputDim, activation: 'linear' })); // Linear activation for Q-values

        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError' // Mean Squared Error is a common loss function for Q-learning
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
            // Load model weights from './bot_model/' directory
            this.model = await tf.loadLayersModel('file://./bot_model/model.json');
            this.targetModel.setWeights(this.model.getWeights()); // Sync target with loaded model
            console.log('[RL_AGENT] Model loaded successfully from file.');
            // When loading a trained model, it's common to start with a lower epsilon
            this.epsilon = this.minEpsilon;
        } catch (e) {
            console.log('[RL_AGENT] No existing model found. Starting with a new model.');
            // console.error('[RL_AGENT] Error loading model:', e); // Uncomment for debugging load issues
        }
    }

    /**
     * Saves the current model weights to the local file system.
     * This should be called periodically or when the bot disconnects.
     */
    async saveModel() {
        try {
            await this.model.save('file://./bot_model');
            console.log('[RL_AGENT] Model saved successfully to file.');
        } catch (e) {
            console.error('[RL_AGENT] Failed to save model:', e);
        }
    }

    /**
     * Stores a single experience (state, action, reward, next state, done) in the
     * experience replay buffer. This buffer is used for batch training to
     * stabilize the learning process.
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
            tf.dispose(stateTensor); // Clean up TensorFlow tensors from memory
            tf.dispose(qValues);
            return action;
        }
    }

    /**
     * Trains the main Q-network using a mini-batch of experiences sampled
     * from the replay buffer. This implements the core of the DQN algorithm.
     */
    async trainModel() {
        // Only train if there are enough experiences in the buffer
        if (this.replayBuffer.length < this.batchSize) {
            return;
        }

        // Periodically update the target Q-network's weights
        this.updateCount++;
        if (this.updateCount % this.targetUpdateFreq === 0) {
            this.targetModel.setWeights(this.model.getWeights());
            console.log('[RL_AGENT] Target network weights updated.');
        }

        // Sample a random mini-batch of experiences from the replay buffer
        // Shuffle and slice to get a random subset
        const batch = this.replayBuffer
            .sort(() => 0.5 - Math.random())
            .slice(0, this.batchSize);

        // Extract components of the experiences
        const states = batch.map(e => e[0]);
        const actions = batch.map(e => e[1]);
        const rewards = batch.map(e => e[2]);
        const nextStates = batch.map(e => e[3]);
        const dones = batch.map(e => e[4]);

        // Use tf.tidy to clean up intermediate tensors automatically
        await tf.tidy(async () => {
            const statesTensor = tf.tensor2d(states);
            const nextStatesTensor = tf.tensor2d(nextStates);

            // Predict Q-values for current states using the main network
            const currentQValues = this.model.predict(statesTensor);
            // Predict Q-values for next states using the target network (for stability)
            const nextQValuesTarget = this.targetModel.predict(nextStatesTensor);

            // Convert current Q-values to a mutable JavaScript array
            const targetQValues = currentQValues.arraySync();

            // Calculate the target Q-value for each experience in the batch
            for (let i = 0; i < this.batchSize; i++) {
                const isDone = dones[i];
                const reward = rewards[i];
                const action = actions[i];

                if (isDone) {
                    // If the episode terminated, the target Q-value is just the immediate reward
                    targetQValues[i][action] = reward;
                } else {
                    // Otherwise, calculate using the Bellman equation: R + gamma * max(Q_target(s', a'))
                    const maxNextQ = tf.max(nextQValuesTarget.slice([i, 0], [1, this.outputDim])).dataSync()[0];
                    targetQValues[i][action] = reward + this.discountFactor * maxNextQ;
                }
            }

            // Convert the calculated target Q-values back to a TensorFlow tensor
            const targetsTensor = tf.tensor2d(targetQValues);

            // Train the main Q-network to predict these target Q-values
            await this.model.fit(statesTensor, targetsTensor, {
                epochs: 1, // Train for a single epoch per batch
                verbose: 0 // Suppress training output
            });
        });

        // Decay epsilon to reduce exploration over time
        this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    }
}


// =============================================================================
// V. Q-AGENT INITIALIZATION
// =============================================================================
// Define the dimensions of the state input vector.
// This must match the number of values returned by `getBotState()`.
const INPUT_NODES = 25; // Increased to accommodate more state variables for advanced goals and memory
// Define the dimension of the action output vector (number of possible actions).
// This must match the number of actions defined in the `ACTIONS` object.
const OUTPUT_NODES = Object.keys(ACTIONS).length;

// Instantiate the Q-Agent with the defined dimensions and hyperparameters
const qAgent = new QAgent(INPUT_NODES, OUTPUT_NODES, {
    learningRate: 0.0005, // Slightly lower learning rate for more stable learning
    epsilonDecay: 0.9998, // Slower epsilon decay for more exploration over longer periods
    replayBufferSize: 250000, // Larger buffer to store more experiences
    batchSize: 128,           // Larger batch size for more stable gradients
    targetUpdateFreq: 500     // Update target network less frequently for stability
});
console.log(`[INIT] Q-Agent initialized with ${INPUT_NODES} inputs and ${OUTPUT_NODES} outputs.`);


// =============================================================================
// VI. BOT STATE REPRESENTATION
// =============================================================================
/**
 * Gathers and normalizes the current state of the bot and its environment
 * into a numerical array suitable as input for the neural network.
 * This function defines the "observation space" of the agent.
 * @returns {number[]} An array of normalized numeric values representing the bot's state.
 */
function getBotState() {
    // Basic Survival Stats (Normalized 0-1)
    const health = bot.health / 20;
    const food = bot.food / 20;
    const inWater = bot.entity.isInWater ? 1 : 0;
    const isDay = bot.time.timeOfDay < 13000 ? 1 : 0; // Approximate day (0-24000 ticks)

    // Enemy Presence (Normalized)
    const nearestHostile = bot.nearestEntity(entity =>
        entity.type === 'mob' && entity.mobType && entity.mobType.isHostile &&
        entity.position.distanceTo(bot.entity.position) < 30
    );
    const hasEnemyNearby = nearestHostile ? 1 : 0;
    const enemyDistance = nearestHostile ? nearestHostile.position.distanceTo(bot.entity.position) / 30 : 0; // 0-1 if within 30 blocks

    // Inventory & Resources (Normalized by max stack or common quantities)
    const hasFoodInInventory = bot.inventory.items().some(item => mcData.foodsArray.some(food => food.id === item.type)) ? 1 : 0;
    const woodCount = bot.inventory.count(mcData.itemsByName.oak_log.id) / 64; // Max stack 64
    const hasWoodenPickaxe = bot.inventory.items().some(item => item.name === 'wooden_pickaxe') ? 1 : 0;
    const stoneCount = bot.inventory.count(mcData.itemsByName.cobblestone.id) / 64;
    const hasStonePickaxe = bot.inventory.items().some(item => item.name === 'stone_pickaxe') ? 1 : 0;
    const coalCount = bot.inventory.count(mcData.itemsByName.coal.id) / 64;
    const ironOreCount = bot.inventory.count(mcData.itemsByName.iron_ore.id) / 64;
    const ironIngotCount = bot.inventory.count(mcData.itemsByName.iron_ingot.id) / 64;
    const hasIronPickaxe = bot.inventory.items().some(item => item.name === 'iron_pickaxe') ? 1 : 0;
    const obsidianCount = bot.inventory.count(mcData.itemsByName.obsidian.id) / 64;
    const hasDiamondPickaxe = bot.inventory.items().some(item => item.name === 'diamond_pickaxe') ? 1 : 0;
    const enderEyesCount = bot.inventory.count(mcData.itemsByName.ender_eye.id) / 12; // Need 12 for portal

    // Game Progression & Location (Binary flags)
    const netherPortalExistsNearby = !!bot.findBlock({
        matching: mcData.blocksByName.nether_portal.id,
        maxDistance: 100 // Check wider area for portals
    });
    const inNether = bot.entity.dimension === 'minecraft:the_nether' ? 1 : 0;
    const inEnd = bot.entity.dimension === 'minecraft:the_end' ? 1 : 0;

    // Basic "Memory" / Historical State (Conceptual for Recurrent RL)
    // A simple flag for recent events. A true RNN would process sequences.
    const recentlyDied = bot.lastDiedTime && (Date.now() - bot.lastDiedTime < 10000) ? 1 : 0; // Died in last 10 seconds

    // Multi-Agent Collaboration State
    const nearestOtherPlayer = bot.nearestEntity(entity => entity.type === 'player' && entity.username !== bot.username);
    const hasOtherPlayerNearby = nearestOtherPlayer ? 1 : 0;
    const otherPlayerDistance = nearestOtherPlayer ? nearestOtherPlayer.position.distanceTo(bot.entity.position) / 60 : 0; // Normalize by 60 blocks

    return [
        health, food, inWater, isDay, hasEnemyNearby, enemyDistance,
        hasFoodInInventory, woodCount, hasWoodenPickaxe, stoneCount,
        hasStonePickaxe, coalCount, ironOreCount, ironIngotCount,
        hasIronPickaxe, obsidianCount, hasDiamondPickaxe, enderEyesCount,
        netherPortalExistsNearby ? 1 : 0, inNether, inEnd,
        recentlyDied, hasOtherPlayerNearby, otherPlayerDistance
    ];
}


// =============================================================================
// VII. ACTION EXECUTION LOGIC
// =============================================================================
/**
 * Executes a specific bot action based on its ID. This function contains the
 * complex logic for multi-step actions and interacts with the Mineflayer API.
 * @param {number} actionId - The ID of the action to execute.
 * @returns {Promise<{success: boolean, actionReward: number}>} Object indicating success and immediate reward.
 */
async function executeBotAction(actionId) {
    console.log(`[ACTION_EXEC] Attempting to execute: ${ACTION_NAMES[actionId]}`);
    let success = false;
    let actionReward = 0; // Immediate reward for this specific action

    try {
        switch (actionId) {
            case ACTIONS.WALK_RANDOM: {
                // Generates a random target location and moves the bot there.
                const randomOffset = Math.random() * 20 - 10; // Offset between -10 and 10 blocks
                const randomX = bot.entity.position.x + randomOffset;
                const randomZ = bot.entity.position.z + randomOffset;
                const randomGoal = new GoalXZ(randomX, randomZ);
                try {
                    await bot.pathfinder.goto(randomGoal);
                    actionReward = 0.1;
                    success = true;
                } catch (err) {
                    console.log(`[ACTION_ERR] Random walk to ${randomX.toFixed(0)}, ${randomZ.toFixed(0)} failed: ${err.message}`);
                    actionReward = -0.1;
                }
                break;
            }

            case ACTIONS.ATTACK_NEAREST_HOSTILE: {
                // Finds the nearest hostile mob and attacks it.
                const nearestHostile = bot.nearestEntity(entity =>
                    entity.type === 'mob' && entity.mobType && entity.mobType.isHostile &&
                    entity.position.distanceTo(bot.entity.position) < 5 // Within attack range
                );
                if (nearestHostile) {
                    bot.setControlState('forward', true); // Move towards target
                    await bot.attack(nearestHostile);
                    bot.setControlState('forward', false);
                    actionReward = 0.5;
                    success = true;
                } else {
                    console.log('[ACTION_INFO] No hostile mob nearby to attack.');
                    actionReward = -0.1; // Small penalty for trying to attack nothing
                }
                break;
            }

            case ACTIONS.FLEE_FROM_HOSTILE: {
                // Calculates a direction away from the nearest hostile mob and moves there.
                const nearestHostile = bot.nearestEntity(entity =>
                    entity.type === 'mob' && entity.mobType && entity.mobType.isHostile
                );
                if (nearestHostile) {
                    const fleeVector = bot.entity.position.minus(nearestHostile.position).normalize().scaled(15);
                    const fleeGoal = new GoalNear(
                        bot.entity.position.x + fleeVector.x,
                        bot.entity.position.y,
                        bot.entity.position.z + fleeVector.z, 5
                    );
                    try {
                        await bot.pathfinder.goto(fleeGoal);
                        actionReward = 0.8;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Flee attempt failed:', err.message);
                        actionReward = -0.5;
                    }
                } else {
                    console.log('[ACTION_INFO] No hostile mob to flee from.');
                    actionReward = -0.1; // Small penalty for attempting to flee from nothing
                }
                break;
            }

            case ACTIONS.BUILD_BASIC_SHELTER: {
                // Attempts to build a simple 3x3x3 dirt shelter around the bot.
                const dirtId = mcData.blocksByName.dirt.id;
                const hasDirt = bot.inventory.count(dirtId);

                if (hasDirt >= 16) { // Enough dirt for base + walls
                    try {
                        await bot.setQuickBarSlot(bot.inventory.items().find(item => item.type === dirtId).slot);
                        const currentPos = bot.entity.position.floor();

                        // Build a 3x3 base (floor)
                        for (let x = -1; x <= 1; x++) {
                            for (let z = -1; z <= 1; z++) {
                                const targetPos = currentPos.offset(x, -1, z); // One block below bot
                                const referenceBlock = bot.blockAt(targetPos);
                                if (referenceBlock && referenceBlock.type !== dirtId && referenceBlock.type !== 0 && bot.canPlaceBlock(referenceBlock)) {
                                    await bot.placeBlock(referenceBlock, vec3(0, 1, 0)); // Place on top of existing block
                                    actionReward += 0.1;
                                    await bot.waitForTicks(5);
                                }
                            }
                        }
                        // Build 3 layers of walls (simple box)
                        for (let y = 0; y < 3; y++) {
                            for (let x = -1; x <= 1; x++) {
                                for (let z = -1; z <= 1; z++) {
                                    // Only place outer walls, not inside bot's space
                                    if (Math.abs(x) === 1 || Math.abs(z) === 1) {
                                        const targetPos = currentPos.offset(x, y, z);
                                        const referenceBlock = bot.blockAt(targetPos.offset(0, -1, 0));
                                        if (referenceBlock && referenceBlock.type !== dirtId && bot.canPlaceBlock(referenceBlock)) {
                                            await bot.placeBlock(referenceBlock, targetPos.minus(referenceBlock.position));
                                            actionReward += 0.15;
                                            await bot.waitForTicks(5);
                                        }
                                    }
                                }
                            }
                        }
                        actionReward += 5.0; // Bonus for attempting shelter
                        success = true;
                        console.log('[ACTION_INFO] Basic dirt shelter constructed.');
                    } catch (err) {
                        console.log('[ACTION_ERR] Building failed:', err.message);
                        actionReward = -1.0;
                    }
                } else {
                    console.log('[ACTION_INFO] Not enough dirt for building (need 16).');
                    actionReward = -0.2; // Penalty for trying without resources
                }
                break;
            }

            case ACTIONS.DIG_WOOD: {
                // Finds the nearest wood log and digs it.
                const woodBlock = bot.findBlock({
                    matching: block => block.name.includes('log') || block.name.includes('wood'),
                    maxDistance: 16
                });
                if (woodBlock && bot.canDigBlock(woodBlock)) {
                    try {
                        await bot.dig(woodBlock);
                        actionReward = 1.5;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Digging wood failed:', err.message);
                        actionReward = -0.5;
                    }
                } else {
                    console.log('[ACTION_INFO] No wood block found nearby to dig.');
                    actionReward = -0.2;
                }
                break;
            }

            case ACTIONS.CRAFT_WOODEN_PICKAXE: {
                // Crafts a wooden pickaxe if materials are available.
                const woodenPickaxeRecipe = bot.recipesFor(mcData.itemsByName.wooden_pickaxe.id).find(r => true);
                if (woodenPickaxeRecipe) {
                    try {
                        await bot.craft(woodenPickaxeRecipe, 1);
                        actionReward = 2.0;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Crafting wooden pickaxe failed:', err.message);
                        actionReward = -1.0;
                    }
                } else {
                    console.log('[ACTION_INFO] No wooden pickaxe recipe available or missing materials.');
                    actionReward = -0.5;
                }
                break;
            }

            case ACTIONS.COLLECT_DROPPED_ITEMS: {
                // Collects the nearest dropped item.
                const itemDrop = bot.nearestEntity(entity => entity.type === 'item' && entity.position.distanceTo(bot.entity.position) < 10);
                if (itemDrop) {
                    try {
                        await bot.collect(itemDrop);
                        actionReward = 0.7;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Collecting item failed:', err.message);
                        actionReward = -0.3;
                    }
                } else {
                    console.log('[ACTION_INFO] No dropped items nearby to collect.');
                    actionReward = -0.1;
                }
                break;
            }

            case ACTIONS.SLEEP: {
                // Attempts to sleep in the nearest bed if it's night.
                const bed = bot.findBlock({
                    matching: block => block.name.includes('bed'),
                    maxDistance: 16,
                    count: 1
                });
                if (bed && !getBotState()[3]) { // If it's night (isDay state is 0)
                    try {
                        await bot.sleep(bed);
                        actionReward = 1.0;
                        success = true;
                        // Wake up after a short period (simulates passing night)
                        setTimeout(async () => { await bot.wake(); }, 5000);
                    } catch (err) {
                        console.log('[ACTION_ERR] Sleeping failed:', err.message);
                        actionReward = -0.5;
                    }
                } else {
                    console.log('[ACTION_INFO] No bed found or not night time.');
                    actionReward = -0.2;
                }
                break;
            }

            case ACTIONS.IDLE: {
                // Bot does nothing, can be a default or explorative action.
                actionReward = 0.05; // Small reward to prevent continuous negative feedback
                success = true;
                break;
            }

            case ACTIONS.MINE_STONE: {
                // Mines the nearest cobblestone block. Requires a wooden pickaxe.
                const stoneBlock = bot.findBlock({
                    matching: mcData.blocksByName.stone.id,
                    maxDistance: 16
                });
                const hasWoodenPick = getBotState()[9]; // Check for wooden pickaxe
                if (stoneBlock && bot.canDigBlock(stoneBlock) && hasWoodenPick) {
                    try {
                        await bot.dig(stoneBlock);
                        actionReward = 1.8;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Mining stone failed:', err.message);
                        actionReward = -0.6;
                    }
                } else {
                    console.log('[ACTION_INFO] No stone found or missing wooden pickaxe.');
                    actionReward = -0.3;
                }
                break;
            }

            case ACTIONS.CRAFT_STONE_PICKAXE: {
                // Crafts a stone pickaxe. Requires 3 cobblestone and 2 sticks.
                const stonePickaxeRecipe = bot.recipesFor(mcData.itemsByName.stone_pickaxe.id).find(r => true);
                const hasEnoughStone = getBotState()[10] * 64 >= 3; // Denormalize stone count
                if (stonePickaxeRecipe && hasEnoughStone) {
                    try {
                        await bot.craft(stonePickaxeRecipe, 1);
                        actionReward = 3.0;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Crafting stone pickaxe failed:', err.message);
                        actionReward = -1.2;
                    }
                } else {
                    console.log('[ACTION_INFO] No stone pickaxe recipe or missing materials (3 cobblestone).');
                    actionReward = -0.6;
                }
                break;
            }

            case ACTIONS.MINE_COAL: {
                // Mines the nearest coal ore. Requires a stone pickaxe or better.
                const coalOre = bot.findBlock({
                    matching: mcData.blocksByName.coal_ore.id,
                    maxDistance: 32
                });
                const hasStoneOrBetterPick = getBotState()[11] || getBotState()[15] || getBotState()[17]; // Stone, Iron, or Diamond Pickaxe
                if (coalOre && bot.canDigBlock(coalOre) && hasStoneOrBetterPick) {
                    try {
                        await bot.dig(coalOre);
                        actionReward = 2.5;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Mining coal failed:', err.message);
                        actionReward = -0.8;
                    }
                } else {
                    console.log('[ACTION_INFO] No coal ore found or missing suitable pickaxe.');
                    actionReward = -0.4;
                }
                break;
            }

            case ACTIONS.MINE_IRON: {
                // Mines the nearest iron ore. Requires a stone pickaxe or better.
                const ironOre = bot.findBlock({
                    matching: mcData.blocksByName.iron_ore.id,
                    maxDistance: 32
                });
                const hasStoneOrBetterPick = getBotState()[11] || getBotState()[15] || getBotState()[17];
                if (ironOre && bot.canDigBlock(ironOre) && hasStoneOrBetterPick) {
                    try {
                        await bot.dig(ironOre);
                        actionReward = 4.0;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Mining iron failed:', err.message);
                        actionReward = -1.5;
                    }
                } else {
                    console.log('[ACTION_INFO] No iron ore found or missing suitable pickaxe.');
                    actionReward = -0.5;
                }
                break;
            }

            case ACTIONS.CRAFT_FURNACE: {
                // Crafts a furnace. Requires 8 cobblestone.
                const furnaceRecipe = bot.recipesFor(mcData.itemsByName.furnace.id).find(r => true);
                const hasEnoughCobblestone = getBotState()[10] * 64 >= 8;
                if (furnaceRecipe && hasEnoughCobblestone) {
                    try {
                        await bot.craft(furnaceRecipe, 1);
                        actionReward = 3.5;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Crafting furnace failed:', err.message);
                        actionReward = -1.5;
                    }
                } else {
                    console.log('[ACTION_INFO] No furnace recipe or missing materials (8 cobblestone).');
                    actionReward = -0.7;
                }
                break;
            }

            case ACTIONS.SMELT_IRON: {
                // Attempts to smelt iron ore in a furnace. Requires coal and iron ore.
                // This is a highly simplified conceptual action. A real implementation
                // would involve placing a furnace, interacting with its inventory,
                // and waiting for smelting to complete.
                const hasFurnaceInInventory = bot.inventory.items().some(item => item.name === 'furnace');
                const hasCoal = getBotState()[12] * 64 >= 1;
                const hasRawIron = getBotState()[13] * 64 >= 1;

                if (hasFurnaceInInventory && hasCoal && hasRawIron) {
                    console.log('[ACTION_INFO] Attempting to smelt iron (conceptual).');
                    actionReward = 5.0; // Reward for having all necessary components
                    success = true;
                    // Actual smelting logic would go here, which is complex for Mineflayer
                    // Example: find place for furnace, place furnace, open furnace, put items, wait, take items
                } else {
                    console.log('[ACTION_INFO] Cannot smelt: Missing furnace, coal, or raw iron.');
                    actionReward = -0.8;
                }
                break;
            }

            case ACTIONS.CRAFT_IRON_PICKAXE: {
                // Crafts an iron pickaxe. Requires 3 iron ingots and 2 sticks.
                const ironPickaxeRecipe = bot.recipesFor(mcData.itemsByName.iron_pickaxe.id).find(r => true);
                const hasEnoughIronIngots = getBotState()[14] * 64 >= 3;
                if (ironPickaxeRecipe && hasEnoughIronIngots) {
                    try {
                        await bot.craft(ironPickaxeRecipe, 1);
                        actionReward = 7.0;
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Crafting iron pickaxe failed:', err.message);
                        actionReward = -2.5;
                    }
                } else {
                    console.log('[ACTION_INFO] No iron pickaxe recipe or missing materials (3 iron ingots).');
                    actionReward = -1.0;
                }
                break;
            }

            case ACTIONS.MINE_DIAMONDS: {
                // Mines the nearest diamond ore. Requires an iron or diamond pickaxe.
                const hasIronOrDiamondPick = getBotState()[15] || getBotState()[17]; // Iron Pickaxe or Diamond Pickaxe
                if (hasIronOrDiamondPick) {
                    const diamondOre = bot.findBlock({
                        matching: mcData.blocksByName.diamond_ore.id,
                        maxDistance: 32
                    });
                    if (diamondOre && bot.canDigBlock(diamondOre)) {
                        console.log('[ACTION_INFO] Found diamond ore, attempting to mine.');
                        try {
                            await bot.dig(diamondOre);
                            actionReward = 15.0;
                            success = true;
                        } catch (err) {
                            console.log('[ACTION_ERR] Mining diamonds failed:', err.message);
                            actionReward = -3.0;
                        }
                    } else {
                        console.log('[ACTION_INFO] No diamond ore found nearby.');
                        actionReward = -1.0;
                    }
                } else {
                    console.log('[ACTION_INFO] No suitable pickaxe (iron or diamond) to mine diamonds.');
                    actionReward = -1.5;
                }
                break;
            }

            case ACTIONS.CRAFT_NETHER_PORTAL: {
                // Attempts to craft and light a Nether Portal. Highly conceptual.
                const obsidianCount = getBotState()[16] * 64; // Denormalize obsidian count
                const hasFlintAndSteel = bot.inventory.items().some(item => item.name === 'flint_and_steel');

                if (obsidianCount >= 14 && hasFlintAndSteel) {
                    console.log('[ACTION_INFO] Attempting to craft a Nether Portal (conceptual).');
                    actionReward = 50.0; // High reward for attempting this milestone
                    success = true;
                    // Actual portal building logic would be very complex, involving precise block placement.
                    // For example, placing 4 obsidian on the ground, then 5 up on each side.
                } else {
                    console.log('[ACTION_INFO] Not enough obsidian (need 14) or no flint and steel to craft portal.');
                    actionReward = -2.0;
                }
                break;
            }

            case ACTIONS.GO_TO_NETHER: {
                // Attempts to enter the Nether dimension via a portal.
                const netherPortalBlock = bot.findBlock({
                    matching: mcData.blocksByName.nether_portal.id,
                    maxDistance: 10
                });
                if (netherPortalBlock) {
                    console.log('[ACTION_INFO] Found Nether Portal, attempting to enter.');
                    try {
                        // Move to the portal block to trigger dimension change
                        await bot.pathfinder.goto(new GoalBlock(netherPortalBlock.position.x, netherPortalBlock.position.y, netherPortalBlock.position.z));
                        actionReward = 100.0; // Very high reward for entering Nether
                        success = true;
                    } catch (err) {
                        console.log('[ACTION_ERR] Failed to go to Nether Portal:', err.message);
                        actionReward = -10.0;
                    }
                } else {
                    console.log('[ACTION_INFO] No Nether Portal found nearby to enter.');
                    actionReward = -2.0;
                }
                break;
            }

            case ACTIONS.KILL_DRAGON: {
                // EXTREMELY COMPLEX AND CONCEPTUAL.
                // A real implementation requires finding stronghold, End portal, fighting Ender Dragon.
                console.log('[ACTION_INFO] Attempting to kill the Ender Dragon (conceptual action).');
                if (getBotState()[20] === 1) { // If bot is in The End dimension
                    // Placeholder: Actual complex combat strategy against dragon and crystals here.
                    actionReward = 1000.0; // HUGE reward for this ultimate goal
                    success = true;
                    console.log('[ACTION_INFO] Placeholder: Bot would now execute complex Ender Dragon fight logic.');
                } else {
                    console.log('[ACTION_INFO] Not in The End dimension to kill dragon.');
                    actionReward = -5.0;
                }
                break;
            }

            case ACTIONS.COMMUNICATE_REQUEST_WOOD: {
                // Sends a chat message requesting wood if the bot is low on it.
                const currentWood = getBotState()[8] * 64; // Denormalize wood count
                const hasOtherPlayerNearby = getBotState()[23];

                if (currentWood < 10 && hasOtherPlayerNearby) {
                    const nearestPlayer = bot.nearestEntity(entity => entity.type === 'player' && entity.username !== bot.username);
                    if (nearestPlayer) {
                        bot.chat(`@${nearestPlayer.username} ${bot.username}: I need wood! Can you help?`);
                        actionReward = 0.2;
                        success = true;
                    }
                } else {
                    console.log('[ACTION_INFO] Not needing wood or no other players nearby to request from.');
                    actionReward = -0.1;
                }
                break;
            }

            case ACTIONS.COMMUNICATE_OFFER_WOOD: {
                // Sends a chat message offering wood if the bot has an abundance.
                const currentWood = getBotState()[8] * 64;
                const hasOtherPlayerNearby = getBotState()[23];

                if (currentWood > 30 && hasOtherPlayerNearby) { // If abundant wood
                    const nearestPlayer = bot.nearestEntity(entity => entity.type === 'player' && entity.username !== bot.username);
                    if (nearestPlayer) {
                        bot.chat(`@${nearestPlayer.username} ${bot.username}: I have extra wood if you need some!`);
                        actionReward = 0.2;
                        success = true;
                    }
                } else {
                    console.log('[ACTION_INFO] Not enough wood to offer or no other players nearby.');
                    actionReward = -0.1;
                }
                break;
            }

            case ACTIONS.COMMUNICATE_ENEMY_WARNING: {
                // Sends a chat message warning about nearby hostile mobs.
                const hasEnemy = getBotState()[4]; // Check for nearby enemy
                const hasOtherPlayerNearby = getBotState()[23];

                if (hasEnemy && hasOtherPlayerNearby) {
                    const nearestPlayer = bot.nearestEntity(entity => entity.type === 'player' && entity.username !== bot.username);
                    const enemy = bot.nearestEntity(entity => entity.type === 'mob' && entity.mobType && entity.mobType.isHostile);
                    if (nearestPlayer && enemy) {
                        bot.chat(`@${nearestPlayer.username} ${bot.username}: Warning! Hostile mob (${enemy.displayName}) at my location: ${enemy.position.x.toFixed(0)}, ${enemy.position.y.toFixed(0)}, ${enemy.position.z.toFixed(0)}!`);
                        actionReward = 0.5;
                        success = true;
                    }
                } else {
                    console.log('[ACTION_INFO] No enemy to warn about or no other players nearby.');
                    actionReward = -0.1;
                }
                break;
            }

            default: {
                // Fallback for unknown actions.
                console.log(`[ACTION_ERR] Unknown action ID: ${actionId}`);
                actionReward = -0.05;
                break;
            }
        }
    } catch (e) {
        // Catch any errors during action execution to prevent bot crash
        console.error(`[ACTION_EXCEPTION] Exception during action ${ACTION_NAMES[actionId]}: ${e.message}`);
        actionReward = -0.7; // Severe penalty for unhandled errors during action
    }

    return { success, actionReward };
}


// =============================================================================
// VIII. REWARD FUNCTION
// =============================================================================
/**
 * Calculates the reward received by the bot based on the transition from an
 * old state to a new state after taking an action. This is crucial for
 * guiding the RL agent's learning process.
 * Reward shaping is used to encourage desired behaviors and discourage undesirable ones.
 * @param {Array<number>} oldState - The state of the bot before the action.
 * @param {Array<number>} newState - The state of the bot after the action.
 * @param {boolean} actionSuccess - Indicates if the executed action was successful.
 * @returns {number} The calculated reward value.
 */
function calculateReward(oldState, newState, actionSuccess) {
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
    if (newState[23] === 1 && oldState[23] === 0) { reward += 0.5; } // Positive reward for another player entering range
    // Specific communication actions receive small immediate rewards in executeBotAction

    // --- Penalty for Failed Actions ---
    if (!actionSuccess) {
        reward -= 1.0; // Penalty for an action that did not succeed
    }

    return reward;
}


// =============================================================================
// IX. MAIN BOT THINKING LOOP (RL EPISODE MANAGEMENT)
// =============================================================================
let lastState = null;     // Stores the state from the previous tick
let lastAction = null;    // Stores the action taken in the previous tick
let currentEpisodeReward = 0; // Tracks total reward for the current episode
let consecutiveIdleActions = 0; // Counter to prevent bot from getting stuck idling

/**
 * The main function where the bot perceives its environment, makes decisions,
 * executes actions, and learns from the outcome. This function is called
 * repeatedly by `setInterval`.
 */
async function botThink() {
    // Ensure Minecraft data is loaded before proceeding
    if (!mcData) {
        console.log('[BOT_THINK] Waiting for Minecraft data to initialize...');
        return;
    }

    const currentState = getBotState(); // Observe the current state of the environment
    console.log(`[STATE] H:${currentState[0].toFixed(2)} F:${currentState[1].toFixed(2)} E:${currentState[4]} W:${currentState[8].toFixed(2)} S:${currentState[9].toFixed(2)} I:${currentState[13].toFixed(2)} DP:${currentState[17]} NP:${currentState[18]} N:${currentState[19]} E:${currentState[20]} Died:${currentState[21]} Player:${currentState[22]} Dist:${currentState[23].toFixed(2)}`);

    let done = false; // Flag to indicate if the current episode has ended

    // If this is not the first tick in an episode, process the previous experience
    if (lastState !== null && lastAction !== null) {
        // 1. Calculate the reward received from the previous action and state transition
        const reward = calculateReward(lastState, currentState, true); // Assuming action was successful
        currentEpisodeReward += reward; // Accumulate reward for the current episode
        console.log(`[REWARD] Previous action reward: ${reward.toFixed(2)}. Total Episode Reward: ${currentEpisodeReward.toFixed(2)}`);

        // 2. Store the (state, action, reward, next_state, done) tuple in the replay buffer
        qAgent.remember(lastState, lastAction, reward, currentState, done);

        // 3. Train the Q-Network model using a batch from the replay buffer
        await qAgent.trainModel();
    }

    // 4. Choose the next action based on the current state using the Q-agent's policy
    const actionId = qAgent.chooseAction(currentState);
    console.log(`[DECISION] Bot chose action: ${ACTION_NAMES[actionId]} (Epsilon: ${qAgent.epsilon.toFixed(4)})`);

    // Monitor if the bot is stuck idling too much, and temporarily increase exploration
    if (actionId === ACTIONS.IDLE) {
        consecutiveIdleActions++;
        if (consecutiveIdleActions > 15 && qAgent.epsilon < 0.1) {
            console.log('[BOT_INFO] Bot idling excessively, temporarily boosting exploration.');
            qAgent.epsilon = 0.3; // Temporarily increase epsilon to encourage new actions
            consecutiveIdleActions = 0; // Reset counter
        }
    } else {
        consecutiveIdleActions = 0; // Reset counter if not idling
    }

    // 5. Execute the chosen action in the Minecraft environment
    const { success, actionReward: immediateActionReward } = await executeBotAction(actionId);

    // Update `lastState` and `lastAction` for the next iteration
    lastState = currentState;
    lastAction = actionId;

    // Check for episode termination conditions (e.g., bot dies)
    if (bot.health <= 0) {
        console.log('[GAME_OVER] Bot died! Episode terminated.');
        done = true; // Mark episode as done
        // Give a significant negative reward for death and train on it immediately
        qAgent.remember(lastState, lastAction, -100, currentState, true);
        await qAgent.trainModel();

        console.log(`[EPISODE_SUMMARY] Episode ended. Final reward: ${currentEpisodeReward.toFixed(2)}`);
        currentEpisodeReward = 0; // Reset episode reward for the next run
        bot.lastDiedTime = Date.now(); // Record death time for the `recentlyDied` state variable

        await qAgent.saveModel(); // Save the learned model weights

        // In a production RL setup, you would typically respawn the bot or reset the environment here.
        // For this script, we'll exit to allow manual restart or external management.
        process.exit(0);
    }
}


// =============================================================================
// X. BOT EVENT HANDLERS
// =============================================================================
// Triggered when the bot successfully connects to the server and spawns in the world.
bot.on('spawn', async () => {
    console.log(`[BOT_EVENT] Bot ${bot.username} spawned into the world!`);
    bot.chat('Hello world! I am an AI bot learning to survive and thrive.');

    // Start the bot's main thinking loop after an initial delay
    // This allows the bot to fully initialize and observe its starting state.
    setTimeout(() => {
        console.log('[BOT_EVENT] Starting bot thinking cycle (every 1 second)...');
        setInterval(botThink, 1000); // Calls botThink() every 1000 milliseconds (1 second)
    }, 5000); // Initial 5-second delay
});

// Triggered when the bot disconnects from the server.
bot.on('end', (reason) => {
    console.log(`[BOT_EVENT] Bot disconnected: ${reason}`);
    qAgent.saveModel(); // Attempt to save the learned model upon disconnection
});

// Triggered when an error occurs in the bot's operation.
bot.on('error', (err) => {
    console.error(`[BOT_EVENT] An unhandled error occurred: ${err.message}`);
});

// Basic event listeners for health and food changes (can be used for debugging)
// bot.on('health', () => { console.log(`[BOT_EVENT] Health: ${bot.health}`); });
// bot.on('food', () => { console.log(`[BOT_EVENT] Food: ${bot.food}`); });

// Listen for chat messages from other players/bots for communication and cooperation.
bot.on('chat', async (username, message) => {
    if (username === bot.username) return; // Ignore messages from self to prevent loops

    console.log(`[CHAT_RX] Received message from ${username}: "${message}"`);

    // Simple message parsing for cooperative actions (Multi-Agent RL conceptual base)
    if (message.includes('Need wood!')) {
        const woodCount = bot.inventory.count(mcData.itemsByName.oak_log.id);
        if (woodCount > 10) {
            bot.chat(`@${username} ${bot.username}: I have ${woodCount} wood! I can share some at my location.`);
            // In a full MARL system, bots would coordinate movement/drop items.
        } else {
            bot.chat(`@${username} ${bot.username}: I don't have much wood, but I'll try to find some.`);
        }
    } else if (message.includes('Have extra wood!')) {
        // If a bot offers wood, this bot could potentially navigate to collect it.
        bot.chat(`@${username} ${bot.username}: Thanks for the offer! I might come to collect if I need it.`);
    } else if (message.includes('Warning! Hostile mob')) {
        // Upon receiving an enemy warning, the bot might prioritize defensive actions.
        console.log(`[COOP_ACTION] Received enemy warning from ${username}. Prioritizing safety.`);
        const currentState = getBotState();
        if (currentState[0] < 0.5) { // If health is low, strongly consider fleeing
            await executeBotAction(ACTIONS.FLEE_FROM_HOSTILE);
        } else { // Otherwise, prepare for attack or investigate (if not already fleeing)
            if (currentState[4] === 0) { // If not already aware of an enemy
                await executeBotAction(ACTIONS.ATTACK_NEAREST_HOSTILE); // Or a "seek enemy" action
            }
        }
    }
});
