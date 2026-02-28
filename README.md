# Rage Against The Machine

## Project Synopsis (Pre-Project)
We want to build 'Rage Against The Machine'—a live brain-computer interface Pong game where a human brain plays against a GPT-powered AI opponent. Using OpenBCI hardware, EEG signals control the player's paddle while a stress-detection layer reads frustration in real-time and makes the AI harder the more you struggle. The only way to win is to calm one's mind. The AI opponent also trash-talks the player using GPT models fed with their live brain state. We have a member with extensive BCI experience, and plan to use Codex to rapidly ship the ship, EEG visualization, signal bridge, and GPT opponent integration. We envision the final demo being an actual game of pong played against AI using just the brain while attempting to master one's mental state.

## Architectural Plan

This system will consist of a few CORE components:

1. Frontend Game (2D Pong Game with 1 DOF OR 3D Atari Combat Game with 2 DOF)
2. EEG Streaming Module (Implemented with reference to [Thonk](https://github.com/aether-raid/thonk))
3. Motor Imagery BCI Classification Module (Implemented with reference to [NeuralFlight](https://github.com/dronefreak/NeuralFlight), adapted to also classify using Muse via [`muse-lsl`](https://github.com/alexandrebarachant/muse-lsl)
4. Opponent Module (Programmatic Opponent + Ragebaiting Mode via Command Centre Signals inspired from [`brain-arcade`](https://github.com/itayinbarr/brain-arcade)




