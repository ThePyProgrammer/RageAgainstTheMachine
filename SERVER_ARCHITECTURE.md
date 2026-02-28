# SERVER_ARCHITECTURE.md

Node.js + TypeScript server with Socket.IO.

Location: `/apps/server/src/`

## Technology Choice: Node.js + Express

**Rationale**:
- Fast iteration with TypeScript
- Socket.IO simplifies WebSocket handling
- Easy deployment (Vercel, Heroku, Railway)
- Shared types with client via monorepo

**Not using Next.js**:
- No SSR needed (single-page game)
- Separate client/server for clearer boundaries
- Lower latency (no framework overhead)

## Server Structure

```
/apps/server/src/
├── index.ts                 # Entry point
├── server.ts                # Express + Socket.IO setup
├── handlers/
│   ├── calibrationHandler.ts
│   ├── gameHandler.ts
│   └── connectionHandler.ts
├── game/
│   ├── gameManager.ts       # Authoritative game state
│   ├── aiManager.ts         # AI difficulty adaptation
│   └── gameEventHandler.ts  # Event triggers
├── calibration/
│   ├── calibrator.ts        # Calibration algorithm
│   └── calibrationService.ts # Session management
├── stress/
│   ├── stressCalculator.ts  # Stress metric
│   └── stressEvents.ts      # Event detection
├── taunts/
│   ├── tauntGenerator.ts    # OpenAI integration
│   └── tauntService.ts      # Rate limiting
└── utils/
    ├── logger.ts
    └── validation.ts
```

## Entry Point

File: `/apps/server/src/index.ts`

```typescript
import { createServer } from './server';

const PORT = process.env.PORT || 3001;

const server = createServer();

server.listen(PORT, () => {
  console.log(`[Server] Listening on port ${PORT}`);
});
```

## Server Setup

File: `/apps/server/src/server.ts`

```typescript
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { ServerToClientEvents, ClientToServerEvents } from '@packages/shared';
import { setupConnectionHandler } from './handlers/connectionHandler';
import cors from 'cors';

export function createServer() {
  const app = express();
  const httpServer = createServer(app);

  // CORS
  app.use(cors({ origin: process.env.CLIENT_URL || 'http://localhost:5173' }));

  // Socket.IO
  const io = new Server<ClientToServerEvents, ServerToClientEvents>(httpServer, {
    cors: {
      origin: process.env.CLIENT_URL || 'http://localhost:5173',
      methods: ['GET', 'POST'],
    },
    pingTimeout: 60000,
    pingInterval: 25000,
  });

  // Connection handler
  setupConnectionHandler(io);

  // Health check
  app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: Date.now() });
  });

  return httpServer;
}
```

## Connection Handler

File: `/apps/server/src/handlers/connectionHandler.ts`

```typescript
import { Server, Socket } from 'socket.io';
import { v4 as uuidv4 } from 'uuid';
import { setupCalibrationHandlers } from './calibrationHandler';
import { setupGameHandlers } from './gameHandler';
import { CalibrationService } from '../calibration/calibrationService';
import { GameManager } from '../game/gameManager';
import { TauntService } from '../taunts/tauntService';
import { ServerEvents } from '@packages/shared';

interface SessionData {
  sessionId: string;
  gameManager: GameManager;
  calibrationService: CalibrationService;
}

const sessions = new Map<string, SessionData>();

export function setupConnectionHandler(io: Server) {
  const calibrationService = new CalibrationService();
  const tauntService = new TauntService(process.env.OPENAI_API_KEY || '');

  io.on('connection', (socket: Socket) => {
    const sessionId = uuidv4();
    console.log(`[Connection] New session: ${sessionId}`);

    // Create session
    const gameManager = new GameManager(sessionId, tauntService, socket);
    sessions.set(sessionId, { sessionId, gameManager, calibrationService });

    // Send session created event
    socket.emit(ServerEvents.SESSION_CREATED, {
      sessionId,
      calibrationRequired: true,
    });

    // Setup handlers
    setupCalibrationHandlers(socket, calibrationService, sessionId);
    setupGameHandlers(socket, gameManager, sessionId);

    // Disconnect
    socket.on('disconnect', () => {
      console.log(`[Connection] Session ended: ${sessionId}`);
      gameManager.cleanup();
      sessions.delete(sessionId);
    });
  });
}
```

## Game Manager

File: `/apps/server/src/game/gameManager.ts`

```typescript
import { GameState, createInitialState, tick } from '@packages/game';
import { StressCalculator } from '../stress/stressCalculator';
import { StressEventDetector } from '../stress/stressEvents';
import { AIManager } from './aiManager';
import { GameEventHandler } from './gameEventHandler';
import { Socket } from 'socket.io';
import { TauntService } from '../taunts/tauntService';
import { ServerEvents, ClientEvents } from '@packages/shared';

export class GameManager {
  private gameState: GameState;
  private stressCalculator = new StressCalculator();
  private stressEventDetector = new StressEventDetector();
  private aiManager = new AIManager();
  private eventHandler: GameEventHandler;
  private gameLoopInterval: NodeJS.Timeout | null = null;
  private lastPlayerCommand: 'UP' | 'DOWN' | 'NEUTRAL' = 'NEUTRAL';
  private rallyCount = 0;

  constructor(
    private sessionId: string,
    tauntService: TauntService,
    private socket: Socket
  ) {
    this.gameState = createInitialState();
    this.eventHandler = new GameEventHandler(tauntService);
  }

  startGame(): void {
    this.gameState.status = 'playing';
    this.gameLoopInterval = setInterval(() => this.gameLoop(), 1000 / 60); // 60 Hz
  }

  onInputCommand(command: 'UP' | 'DOWN' | 'NEUTRAL', activation: number): void {
    this.lastPlayerCommand = command;

    // Update stress metric
    this.stressCalculator.addSample({ activation, command, timestamp: Date.now() });
  }

  private gameLoop(): void {
    // Update stress (throttled to once per second)
    const stressLevel = this.stressCalculator.computeStress();
    const stressEvents = this.stressEventDetector.detectEvents(stressLevel);

    // Update AI difficulty
    this.aiManager.updateDifficulty(stressLevel);
    const aiCommand = this.aiManager.calculateCommand(this.gameState);

    // Tick game
    const prevState = this.gameState;
    this.gameState = tick(this.gameState, this.lastPlayerCommand, aiCommand);

    // Detect game events
    this.detectGameEvents(prevState, this.gameState, stressLevel);

    // Handle stress events
    stressEvents.forEach((event) => {
      this.eventHandler.onGameEvent(event.type, {
        stressLevel,
        score: this.gameState.score,
        rallyLength: this.rallyCount,
      }, this.socket);
    });

    // Broadcast state
    this.socket.emit(ServerEvents.GAME_STATE, {
      timestamp: Date.now(),
      ball: this.gameState.ball,
      playerPaddle: this.gameState.playerPaddle,
      aiPaddle: this.gameState.aiPaddle,
      score: this.gameState.score,
      stressLevel,
    });

    // Check game over
    if (this.gameState.status === 'game_over') {
      this.endGame();
    }
  }

  private detectGameEvents(prev: GameState, current: GameState, stressLevel: number): void {
    // Score change
    if (current.score.player > prev.score.player) {
      this.eventHandler.onGameEvent('player_score', {
        stressLevel,
        score: current.score,
        rallyLength: this.rallyCount,
      }, this.socket);
      this.rallyCount = 0;
    } else if (current.score.ai > prev.score.ai) {
      this.eventHandler.onGameEvent('ai_score', {
        stressLevel,
        score: current.score,
        rallyLength: this.rallyCount,
      }, this.socket);
      this.rallyCount = 0;
    }

    // Rally tracking
    // (Simplified: increment on paddle hit, detect in collision logic)
    if (this.rallyCount > 15) {
      this.eventHandler.onGameEvent('long_rally', {
        stressLevel,
        score: current.score,
        rallyLength: this.rallyCount,
      }, this.socket);
      this.rallyCount = 0; // Reset after taunt
    }
  }

  private endGame(): void {
    if (this.gameLoopInterval) {
      clearInterval(this.gameLoopInterval);
      this.gameLoopInterval = null;
    }

    this.socket.emit(ServerEvents.GAME_OVER, {
      winner: this.gameState.winner!,
      finalScore: this.gameState.score,
    });
  }

  cleanup(): void {
    if (this.gameLoopInterval) {
      clearInterval(this.gameLoopInterval);
    }
  }
}
```

## Environment Variables

File: `/apps/server/.env.example`

```
PORT=3001
CLIENT_URL=http://localhost:5173
OPENAI_API_KEY=sk-...
NODE_ENV=development
```

## Deployment

### Docker

File: `/apps/server/Dockerfile`

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY dist ./dist
EXPOSE 3001
CMD ["node", "dist/index.js"]
```

### Railway / Heroku

```bash
# Build command
npm run build

# Start command
npm run start
```

## Testing

- [ ] Server starts without errors
- [ ] WebSocket connection established
- [ ] Session created event sent
- [ ] Game loop runs at 60 Hz
- [ ] Stress metric updates
- [ ] AI difficulty adapts
- [ ] Taunts sent correctly
