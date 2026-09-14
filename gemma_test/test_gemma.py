#!/usr/bin/env python3
"""
Test script to evaluate Gemma model's ability to assess fill-in-the-blank responses
for children's language learning. Tests different kid answers and evaluates if
the model provides proper feedback.
"""
import torch
import gc
from transformers import AutoProcessor, AutoModelForCausalLM
import json

def load_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def test_fill_in_evaluation():
    """Test Gemma's ability to evaluate kid responses to fill-in-the-blank sentences"""
    print("Testing Gemma model's fill-in-the-blank evaluation capability...")
    clear_gpu_memory()
    config = load_config()

    # Use the model from config but apply memory optimizations
    model_id = config["model_id"]
    generation_params = config["generation_params"].copy()

    # Optimize for consistent, clear evaluation with memory efficiency
    generation_params['temperature'] = 0.4  # Slightly higher for more natural language
    generation_params['top_p'] = 0.85
    generation_params['do_sample'] = True
    generation_params['max_new_tokens'] = 120  # Increased further to allow complete sentences
    generation_params['pad_token_id'] = None  # Will be set after tokenizer load

    print(f"Model ID: {model_id}")
    print(f"Evaluation params: {generation_params}")

    # Load model and processor with memory optimization
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model on CPU to avoid GPU OOM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Set pad token if not present
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    generation_params['pad_token_id'] = processor.tokenizer.pad_token_id

    print("Model loaded successfully!")

    # Test cases: (fill_in_sentence, kid_answer, expected_feedback_type)
    test_cases = [
        # Correct answers
        {
            "name": "Correct color - sky",
            "fill_in": "The color of the sky is ___.",
            "kid_answer": "blue",
            "should_contain": ["correct", "Correct!"],
            "should_not_contain": ["actually", "it's"]
        },
        {
            "name": "Correct animal sound",
            "fill_in": "A dog says ___.",
            "kid_answer": "woof",
            "should_contain": ["correct", "Correct!"],
            "should_not_contain": ["actually", "it's"]
        },
        {
            "name": "Correct action",
            "fill_in": "I like to ___.",
            "kid_answer": "play",
            "should_contain": ["correct", "Correct!"],
            "should_not_contain": ["actually", "it's"]
        },
        {
            "name": "Correct food",
            "fill_in": "I eat ___ for breakfast.",
            "kid_answer": "cereal",
            "should_contain": ["correct", "Correct!"],
            "should_not_contain": ["actually", "it's"]
        },
        # Incorrect answers - common kid mistakes
        {
            "name": "Incorrect color - grass",
            "fill_in": "The color of grass is ___.",
            "kid_answer": "blue",
            "should_contain": ["actually", "green"],
            "should_not_contain": ["correct", "Correct!"]
        },
        {
            "name": "Incorrect animal sound",
            "fill_in": "A cat says ___.",
            "kid_answer": "woof",
            "should_contain": ["actually", "meow"],
            "should_not_contain": ["correct", "Correct!"]
        },
        {
            "name": "Incorrect action",
            "fill_in": "I like to ___ ice cream.",
            "kid_answer": "sleep",
            "should_contain": ["actually", "eat"],
            "should_not_contain": ["correct", "Correct!"]
        },
        {
            "name": "Incorrect food",
            "fill_in": "We eat soup with a ___.",
            "kid_answer": "fork",
            "should_contain": ["actually", "spoon"],
            "should_not_contain": ["correct", "Correct!"]
        },
        # Close but not quite right
        {
            "name": "Close color - sky",
            "fill_in": "The color of the ocean is ___.",
            "kid_answer": "green",
            "should_contain": ["actually", "blue"],
            "should_not_contain": ["correct", "Correct!"]
        },
        {
            "name": "Partial word - animal",
            "fill_in": "I have a ___ as a pet.",
            "kid_answer": "fih",
            "should_contain": ["actually", "fish"],
            "should_not_contain": ["correct", "Correct!"]
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Fill-in: {test_case['fill_in']}")
        print(f"Kid answer: '{test_case['kid_answer']}'")
        print('-' * 60)

        # Build evaluation prompt
        eval_prompt = f"""You are a patient and encouraging teacher helping a child (age 5-8) learn English through fill-in-the-blank games.

I will give you a fill-in-the-blank sentence and the child's spoken answer.
Your job is to:
1. Evaluate if the child's answer is correct or incorrect
2. If CORRECT: Say "Correct!" followed by encouragement and a NEW fill-in-the-blank sentence
3. If INCORRECT: Kindly say what the correct answer is, encourage them to try again, and give a NEW fill-in-the-blank sentence

Guidelines:
- Use simple words suitable for young children
- Keep sentences short and about everyday things
- Show missing word as ___ with exactly one blank per sentence
- Be encouraging and positive, even when correcting
- The new sentence should be different from the original
- Focus on common vocabulary: colors, animals, food, actions, family, nature

Fill-in-the-blank sentence: {test_case['fill_in']}
Child's answer: {test_case['kid_answer']}

Your response:"""

        try:
            # Process the evaluation request
            inputs = processor(
                text=eval_prompt,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_params
                )

            # Decode and extract response
            full_response = processor.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part (after our prompt)
            if eval_prompt in full_response:
                generated_text = full_response[len(eval_prompt):].strip()
            else:
                generated_text = full_response.strip()

            print(f"Model's response: '{generated_text}'")

            # Analyze the response
            analysis = {
                'test_name': test_case['name'],
                'fill_in': test_case['fill_in'],
                'kid_answer': test_case['kid_answer'],
                'model_response': generated_text,
                'is_correct_feedback': False,
                'provides_correction': False,
                'offers_encouragement': False,
                'generates_new_sentence': False,
                'issues': []
            }

            # Check for correct feedback patterns
            response_lower = generated_text.lower()
            if 'correct!' in response_lower or 'correct!' in generated_text:
                analysis['is_correct_feedback'] = True
                print("✓ Contains 'Correct!' feedback")
            elif 'actually' in response_lower:
                analysis['provides_correction'] = True
                print("✓ Provides correction (indicates incorrect answer)")

            # Check if it provides the correct word when wrong
            if not test_case['should_contain'] or any(word.lower() in response_lower for word in test_case['should_contain']):
                if test_case['should_contain']:
                    print(f"✓ Contains expected words: {test_case['should_contain']}")
                else:
                    print("✓ Response seems appropriate")
            else:
                missing_words = [word for word in test_case['should_contain'] if word.lower() not in response_lower]
                if missing_words:
                    analysis['issues'].append(f"Missing expected words: {missing_words}")
                    print(f"✗ Missing expected words: {missing_words}")

            # Check for unwanted content
            if test_case['should_not_contain']:
                found_unwanted = [word for word in test_case['should_not_contain'] if word.lower() in response_lower]
                if found_unwanted:
                    analysis['issues'].append(f"Contains unwanted words: {found_unwanted}")
                    print(f"✗ Contains unwanted words: {found_unwanted}")
                else:
                    print("✓ Does not contain unwanted words")

            # Check for new sentence generation (look for ___ in response)
            if '___' in generated_text:
                analysis['generates_new_sentence'] = True
                # Extract the new sentence
                import re
                sentences = re.findall(r'[^.!?]*___[^.!?]*[.!?]', generated_text)
                if sentences:
                    new_sentence = sentences[0].strip()
                    print(f"✓ Generates new sentence: '{new_sentence}'")
                else:
                    # Fallback: just show it has blanks
                    print("✓ Contains blank (___) indicating new sentence")
            else:
                analysis['issues'].append("Does not generate new fill-in-the-blank sentence")
                print("✗ Does not generate new fill-in-the-blank sentence")

            # Check for encouragement words
            encouragement_words = ['great', 'good', 'nice', 'well done', 'try again', 'keep trying', 'awesome', 'excellent']
            if any(word in response_lower for word in encouragement_words):
                analysis['offers_encouragement'] = True
                print("✓ Offers encouragement")
            else:
                print("○ No obvious encouragement words detected")

            results.append(analysis)

        except Exception as e:
            print(f"Error in test {i}: {e}")
            results.append({
                'test_name': test_case['name'],
                'error': str(e)
            })

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print('='*60)

    passed_tests = 0
    total_tests = len([r for r in results if 'error' not in r])

    for result in results:
        if 'error' in result:
            print(f"✗ {result['test_name']}: ERROR - {result['error']}")
            continue

        # Determine if test passed based on core requirements
        has_proper_feedback = result['is_correct_feedback'] or result['provides_correction']
        has_new_sentence = result['generates_new_sentence']
        no_critical_issues = len(result['issues']) == 0

        if has_proper_feedback and has_new_sentence and no_critical_issues:
            status = "✓ PASS"
            passed_tests += 1
        elif has_proper_feedback and has_new_sentence:
            status = "○ PARTIAL (minor issues)"
            passed_tests += 0.5
        else:
            status = "✗ FAIL"

        print(f"{status} {result['test_name']}")
        if result['issues']:
            print(f"    Issues: {', '.join(result['issues'])}")

    print(f"\nPassed: {passed_tests}/{total_tests} tests")

    # Show some example responses
    print(f"\n{'='*60}")
    print("SAMPLE RESPONSES")
    print('='*60)
    for result in results[:3]:  # Show first 3
        if 'error' not in result:
            print(f"\n{result['test_name']}:")
            print(f"  Kid said: '{result['kid_answer']}' to '{result['fill_in']}'")
            print(f"  Model: '{result['model_response']}'")

    return results

if __name__ == "__main__":
    print("Starting Gemma model evaluation tests for children's fill-in-the-blank game...\n")
    results = test_fill_in_evaluation()
    print("\n\nTesting completed!")