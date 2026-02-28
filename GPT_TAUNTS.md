# GPT_TAUNTS.md

Generate ragebaiting trash-talk speech bubbles via GPT during key game events.

Location: `/apps/server/src/taunts/`

## Event Triggers

Defined in `/packages/shared/src/events.ts`:

```typescript
export type TauntTrigger =
  | 'player_miss'
  | 'player_score'
  | 'ai_score'
  | 'long_rally'
  | 'stress_spike'
  | 'calm_streak';
```

### Trigger Conditions

| Trigger | Condition | Example Taunt |
|---------|-----------|---------------|
| `player_miss` | Player fails to hit ball | "Is that all you got?" |
| `player_score` | Player scores | "Lucky shot!" |
| `ai_score` | AI scores | "Too easy!" |
| `long_rally` | > 15 paddle hits | "Getting tired yet?" |
| `stress_spike` | Stress jumps > 0.3 | "Feeling the pressure?" |
| `calm_streak` | Stress < 0.3 for 10s | "Bored already?" |

## GPT Integration

File: `/apps/server/src/taunts/tauntGenerator.ts`

```typescript
import OpenAI from 'openai';

export interface TauntContext {
  trigger: TauntTrigger;
  stressLevel: number;
  score: { player: number; ai: number };
  rallyLength: number; // Number of paddle hits
}

export class TauntGenerator {
  private openai: OpenAI;
  private systemPrompt = `You are a trash-talking Pong AI opponent. Generate short, snarky, ragebaiting comments (max 60 characters) based on game events. Be provocative but not offensive. Use gaming slang and memes.`;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey });
  }

  /**
   * Generate taunt from GPT
   */
  async generate(context: TauntContext): Promise<string> {
    const userPrompt = this.buildPrompt(context);

    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        max_tokens: 30,
        temperature: 0.9, // High creativity
        timeout: 5000, // 5s timeout
      });

      const taunt = response.choices[0]?.message?.content?.trim() || this.getFallback(context.trigger);
      return taunt.slice(0, 60); // Enforce max length
    } catch (error) {
      console.error('[TauntGenerator] GPT error:', error);
      return this.getFallback(context.trigger);
    }
  }

  /**
   * Build contextual prompt
   */
  private buildPrompt(context: TauntContext): string {
    const { trigger, stressLevel, score, rallyLength } = context;

    const stressDesc =
      stressLevel > 0.7 ? 'very stressed' : stressLevel > 0.4 ? 'stressed' : 'calm';

    switch (trigger) {
      case 'player_miss':
        return `Player missed the ball. They are ${stressDesc}. Score: ${score.player}-${score.ai}. Taunt them.`;
      case 'player_score':
        return `Player scored. They are ${stressDesc}. Score: ${score.player}-${score.ai}. Downplay their win.`;
      case 'ai_score':
        return `You (AI) scored. Player is ${stressDesc}. Score: ${score.player}-${score.ai}. Gloat.`;
      case 'long_rally':
        return `Long rally (${rallyLength} hits). Player is ${stressDesc}. Taunt their endurance.`;
      case 'stress_spike':
        return `Player's stress spiked. They are ${stressDesc}. Mock their nerves.`;
      case 'calm_streak':
        return `Player is too calm. Provoke them.`;
      default:
        return 'Generic taunt.';
    }
  }

  /**
   * Fallback taunts (if GPT fails)
   */
  private getFallback(trigger: TauntTrigger): string {
    const fallbacks: Record<TauntTrigger, string[]> = {
      player_miss: ['Oops!', 'Too slow!', 'Try harder!'],
      player_score: ['Lucky!', 'Flukey!', "Won't happen again!"],
      ai_score: ['Too easy!', 'Gg ez!', 'Outplayed!'],
      long_rally: ['Tired yet?', 'Keep up!', "Can't finish?"],
      stress_spike: ['Cracking?', 'Feeling it?', 'Stay calm!'],
      calm_streak: ['Bored?', 'Wake up!', 'Try harder!'],
    };

    const options = fallbacks[trigger] || ['...'];
    return options[Math.floor(Math.random() * options.length)];
  }
}
```

## Rate Limiting

File: `/apps/server/src/taunts/tauntService.ts`

```typescript
import { TauntGenerator, TauntContext } from './tauntGenerator';
import { TauntTrigger } from '@packages/shared';

const RATE_LIMIT_MS = 3000; // Max 1 taunt per 3 seconds

export class TauntService {
  private generator: TauntGenerator;
  private lastTauntTime = 0;
  private queue: TauntContext[] = [];

  constructor(apiKey: string) {
    this.generator = new TauntGenerator(apiKey);
  }

  /**
   * Request taunt (rate-limited)
   */
  async requestTaunt(context: TauntContext): Promise<string | null> {
    const now = Date.now();

    // Rate limit check
    if (now - this.lastTauntTime < RATE_LIMIT_MS) {
      // Queue for later (or discard)
      return null;
    }

    this.lastTauntTime = now;

    // Generate taunt
    const taunt = await this.generator.generate(context);
    return taunt;
  }
}
```

## Event Handler Integration

File: `/apps/server/src/game/gameEventHandler.ts`

```typescript
import { TauntService } from '../taunts/tauntService';
import { Socket } from 'socket.io';
import { ServerEvents, TauntTrigger } from '@packages/shared';

export class GameEventHandler {
  private tauntService: TauntService;

  constructor(tauntService: TauntService) {
    this.tauntService = tauntService;
  }

  /**
   * Handle game event and potentially trigger taunt
   */
  async onGameEvent(
    trigger: TauntTrigger,
    context: { stressLevel: number; score: { player: number; ai: number }; rallyLength: number },
    socket: Socket
  ): Promise<void> {
    const taunt = await this.tauntService.requestTaunt({ trigger, ...context });

    if (taunt) {
      socket.emit(ServerEvents.TAUNT_MESSAGE, {
        text: taunt,
        durationMs: 4000, // Display for 4 seconds
        trigger,
      });
    }
  }
}
```

## Client-Side Display

Taunts rendered as speech bubbles (see `GAME_LOOP_AND_RENDERING.md`).

### Styling (Tailwind)

```typescript
// Example component
<div className="fixed top-20 left-1/2 -translate-x-1/2 bg-gray-900 border-2 border-red-500 rounded-lg px-4 py-2 shadow-xl animate-bounce">
  <p className="text-white font-bold text-lg">{tauntText}</p>
</div>
```

## Edge Cases

| Case | Handling |
|------|----------|
| GPT timeout | Use fallback taunt |
| GPT rate limit | Use fallback taunt, log warning |
| Offensive content | Add content filter (OpenAI moderation API) |
| Network lag | Discard old taunts (timestamp check) |
| Rapid events | Queue and throttle to 1 per 3s |

## Testing

### Unit Tests
- [ ] Prompt builder generates correct context
- [ ] Fallback taunts always return valid string
- [ ] Rate limiter enforces 3s minimum gap

### Integration Tests
- [ ] GPT API call succeeds with valid API key
- [ ] Timeout triggers fallback
- [ ] Taunts sent via WebSocket correctly

### Manual Tests
- [ ] Taunts appear on screen
- [ ] Taunts fade after duration
- [ ] Taunts are contextually appropriate
- [ ] No offensive content

## OpenAI API Configuration

```typescript
// Environment variable
OPENAI_API_KEY=sk-...

// Initialize
const tauntService = new TauntService(process.env.OPENAI_API_KEY!);
```

## Provider Abstraction

File: `/apps/server/src/taunts/providers/ITauntProvider.ts`

```typescript
export interface ITauntProvider {
  generate(context: TauntContext): Promise<string>;
}

// OpenAI implementation
export class OpenAITauntProvider implements ITauntProvider {
  // ... (from tauntGenerator.ts)
}

// Mock implementation (for testing)
export class MockTauntProvider implements ITauntProvider {
  async generate(context: TauntContext): Promise<string> {
    return `Mock taunt for ${context.trigger}`;
  }
}
```

Usage:

```typescript
const provider = process.env.NODE_ENV === 'test'
  ? new MockTauntProvider()
  : new OpenAITauntProvider(process.env.OPENAI_API_KEY!);

const tauntService = new TauntService(provider);
```
