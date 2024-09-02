
from openai import OpenAI
import json

# Define API key
key = 'sk-'
client = OpenAI(api_key=key)

# Variables for model evaluations
total_questions = 0
correct_answers = 0

# preprocess the annotation by Expand the acronyms
def preprocess_annotations(data):
    para_anno = data.get("para_anno", {})
    question_anno = data.get("question_anno", {})

    # Handle case where question_anno might be a list
    if isinstance(question_anno, list):
        question_anno_dict = {}
        for item in question_anno:
            if isinstance(item, dict):
                question_anno_dict.update(item)
        question_anno = question_anno_dict

    expanded_annotations = {
        "paragraph_annotations": {
            "effect_direction_sign": para_anno.get("effect_dir_sign", ""),
            "cause_direction_sign": para_anno.get("cause_dir_sign", ""),
            "effect_property": para_anno.get("effect_prop", ""),
            "cause_property": para_anno.get("cause_prop", ""),
            "cause_direction_string": para_anno.get("cause_dir_str", ""),
            "effect_direction_string": para_anno.get("effect_dir_str", "")
        },
        "question_annotations": {
            "more_effect_direction": question_anno.get("more_effect_dir", ""),
            "more_cause_property": question_anno.get("more_cause_prop", ""),
            "more_cause_direction": question_anno.get("more_cause_dir", ""),
            "less_effect_property": question_anno.get("less_effect_prop", ""),
            "less_effect_direction": question_anno.get("less_effect_dir", ""),
            "more_effect_property": question_anno.get("more_effect_prop", "")
        }
    }

    return expanded_annotations

# Read train.jsonl file line by line
with open("C:/Users/najwa/OneDrive/Desktop/PhD/code/openai_quartz/quartz_dataset/train.jsonl", "r") as json_file:
    lines = json_file.readlines()
    
    # Read each line
    for line in lines:
        # Parse the JSON data
        data = json.loads(line)
        
        print("Data:", data)  # Debugging print

        # Get the question and expected answers
        question_stem = data["question"]["stem"]
        choices = data["question"]["choices"]
        expected_answer = data["answerKey"]

        # Preprocess the annotations
        expanded_annotations = preprocess_annotations(data)

        # Debugging print
        print("Question Stem:", question_stem) 
        print("Choices:", choices) 
        print("Expected Answer:", expected_answer)  
        print("Expanded Annotations:", expanded_annotations)  
 

        # Combine question stem, choices, and annotations into a prompt
        para_annotations = "\n".join([f"{key}: {value}" for key, value in expanded_annotations["paragraph_annotations"].items()])
        question_annotations = "\n".join([f"{key}: {value}" for key, value in expanded_annotations["question_annotations"].items()])
        annotations = f"Paragraph Annotations:\n{para_annotations}\n\nQuestion Annotations:\n{question_annotations}\n"
        prompt = question_stem + "\n" + "\n".join([f"{choice['label']}. {choice['text']}" for choice in choices]) + "\n\n" + annotations

        # Debugging print
        print("Prompt:", prompt)  
        
        # Define the API request data
        data_for_request = {
            "model": "gpt-3.5-turbo",
            "prompt": prompt,
            "max_tokens": 50 # increase it to max the length of the justification
        }

        try:
            # Perform the API request
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "answer the question by choose A or B then justifying the right choice. return A or B in JSON form with fields 'Final_Answer' and a justification field 'Explanation' and do not include markdown and write the results in results.jsonl file in JSON form"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the response
            print(completion.choices[0].message.content)
            response = json.loads(completion.choices[0].message.content)
            generated_answer = response.get("Final_Answer", "")
            explanation = response.get("Explanation", "")

            # Create a JSON object with the question, expected answer, generated answer, and annotations
            result_data = {
                "question": question_stem,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "Explanation": explanation,
                "expanded_annotations": expanded_annotations
            }

            print("Writing to file:", result_data)  # Check what is being written

            # Write to results.jsonl file line by line
            with open("C:/Users/najwa/OneDrive/Desktop/PhD/code/openai_quartz/quartz_dataset/resultsgpt3.5 - with context.jsonl", "a") as results_file:
                results_file.write(json.dumps(result_data) + "\n")

            # Counting the correct answer
            total_questions += 1
            if generated_answer == expected_answer:
                correct_answers += 1
        except Exception as e:
            print(f"An error occurred: {e}")

# Calculate the accuracy and printing the results
accuracy = correct_answers / total_questions if total_questions > 0 else 0
print("Total Questions:", total_questions)
print("Correct Answers:", correct_answers)
print("Accuracy:", accuracy)

