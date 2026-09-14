import torch

from transformers import AutoProcessor, AutoModelForCausalLM


class SentenceCoachAgent:

    def __init__(self, model_id, system_prompt, generation_params):

        self.system_prompt = system_prompt
        self.generation_params = generation_params

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def process_turn(self, audio_array, sampling_rate):

        prompt = f"{self.system_prompt}\nUser Audio:\n<|audio|>"

        inputs = self.processor(
            text=prompt,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            **self.generation_params
        )

        response_text = self.processor.decode(outputs[0] , skip_special_tokens=True)

        return response_text.split(self.system_prompt)[-1].strip()