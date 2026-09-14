import json 
from audio_utils import speak, record_audio
from agent import SentenceCoachAgent


def load_config(filepath="config.json"):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

def run_orchestrator():
    config = load_config()

    print("Starting the orchestrator...")

    agent = SentenceCoachAgent(

        model_id = config["model_id"],
        system_prompt = config["system_prompt"],
        generation_params = config["generation_params"]

    )

    start_text = config["start_text"]
    print(f"Agent: {start_text}")
    speak(start_text)

    dur = config["audio_params"]["duration_seconds"]
    sr = config["audio_params"]["sampling_rate"]

    while True:

        try:
            audio_array, _ = record_audio(duration=dur, fs = sr)

            print("Evaluating ....")

            response_text = agent.process_turn(audio_array, sr)

            print(f"Agent: {response_text}")
            speak(response_text)

        except KeyboardInterrupt:
            print("Exiting the orchestrator...")
            break


if __name__ == "__main__":
    run_orchestrator()




