# SuperApp - Fill-in-the-Blank Language Learning System

A real-time language learning application that uses speech recognition and AI to play a fill-in-the-blank game. The user speaks their guess, the system evaluates it, provides feedback, and generates a new sentence for the next round.

## System Architecture

The application follows a hub-and-spoke orchestration model:

1. **User Speech** → Audio Recording (`audio_utils.py`)
2. **Audio Processing** → Speech-to-Text & Evaluation (`agent.py` with Gemma model)
3. **Orchestration** → Main control flow (`main.py`)
4. **Text-to-Speech** → Audio Output (`audio_utils.py`)
5. **Back to User** → Continuous loop

## Components

- `main.py`: Orchestrator that coordinates the flow between components
- `agent.py`: Contains the SentenceCoachAgent using Google's Gemma model for speech evaluation and response generation
- `audio_utils.py`: Handles audio recording (sounddevice) and text-to-speech (pyttsx3)
- `config.json`: Configuration for model, prompts, generation parameters, and audio settings
- `environment.yml`: Conda environment specification for dependencies


## How to Run

1. **Set up the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate sentence_coach
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Usage**:
   - The system will start with an initial fill-in-the-blank sentence
   - Speak your guess when prompted
   - The system will evaluate your response and provide feedback
   - A new sentence will be generated for your next guess
   - Press Ctrl+C to exit

## Requirements

- Conda or Python 3.10+
- Audio input/output device
- Internet connection (for initial model download)

## Configuration

Adjust settings in `config.json`:
- Model parameters and generation settings
- Audio duration and sampling rate
- Initial prompt and system instructions for the AI tutor

## Notes

The system uses Google's Gemma 4B instructive model for natural language understanding and generation. Ensure you have access to this model through Hugging Face or adjust the model_id in config.json if using a different model.