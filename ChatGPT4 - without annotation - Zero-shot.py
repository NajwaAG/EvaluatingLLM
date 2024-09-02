
from openai import OpenAI
import json

# Define API key
key = 'sk-'
client = OpenAI(api_key=key)

# Variables for model evaluations
total_questions = 0
correct_answers = 0

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

        # Extract annotations if available
        para_anno = data.get("para_anno", {})
        question_anno = data.get("question_anno", {})

        # Debugging print
        print("Question Stem:", question_stem) 
        print("Choices:", choices) 
        print("Expected Answer:", expected_answer)  
        print("Para Annotation:", para_anno)  
        print("Question Annotation:", question_anno)  

        # Combine question stem, choices, and annotations into a prompt
        para_annotations = "\n".join([f"{key}: {value}" for key, value in para_anno.items()])

        # Handle the case where question_anno is a list or dictionary
        if isinstance(question_anno, list):
            question_annotations = "\n".join([f"{key}: {value}" for anno in question_anno for key, value in anno.items()])
        else:
            question_annotations = "\n".join([f"{key}: {value}" for key, value in question_anno.items()])

        annotations = f"Paragraph Annotations:\n{para_annotations}\n\nQuestion Annotations:\n{question_annotations}\n"

        prompt = question_stem + "\n" + "\n".join([f"{choice['label']}. {choice['text']}" for choice in choices]) + "\n\n" + annotations

        # Debugging print
        print("Prompt:", prompt)  
        
        # Define the API request data
        data_for_request = {
            "model": "gpt-4-turbo",
            "prompt": prompt,
            "max_tokens": 20
        }

        try:
            # Perform the API request
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "answer the question by choose A or B then justifying the right choice. return A or B in JSON form with fields 'Final_Answer' and a justification field 'Explanation' and do not include markdown and write the results in resultsgpt4.jsonl file in JSON form"},
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
                "para_anno": para_anno,
                "question_anno": question_anno
            }

            print("Writing to file:", result_data)  # Check what is being written

            # Write to results.jsonl file line by line
            with open("C:/Users/najwa/OneDrive/Desktop/PhD/code/openai_quartz/quartz_dataset/resultsgpt4.jsonl", "a") as results_file:
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

