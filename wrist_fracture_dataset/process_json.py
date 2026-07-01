import json
import os
import sys

def process_json(input_file, output_json, output_txt):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract the required fields
    processed_data = {
        "input_file_contents": data.get("input_file_contents", ""),
        "question": data.get("formatted_response", {}).get("question", ""),
        "answer": data.get("formatted_response", {}).get("answer", ""),
        "grounding_phrase": data.get("formatted_response", {}).get("grounding_phrase", "")
    }
    
    # Write the processed data to the JSON output file
    with open(output_json, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Write the processed data to the text output file
    with open(output_txt, 'w') as f:
        for key, value in processed_data.items():
            f.write(f"{key}:\n{value}\n")
            f.write('*' * 50 + '\n')  # Separator line

def process_directory(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith("_report_v8.json"):
            input_file = os.path.join(directory, filename)
            output_json = os.path.join(directory, filename.replace("_report_v8.json", "_report_v8_short.json"))
            output_txt = os.path.join(directory, filename.replace("_report_v8.json", "_report_v8_short.txt"))
            process_json(input_file, output_json, output_txt)
            print(f"Processed {filename} -> {os.path.basename(output_json)} and {os.path.basename(output_txt)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_json.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    process_directory(directory)
    print("Processing complete.")