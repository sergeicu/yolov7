import re
import sys
import json

def extract_core_text(content, start_markers, end_markers):
    for start_marker in start_markers:
        for end_marker in end_markers:
            pattern = f'{re.escape(start_marker)}\\s*(.*?)\\s*{re.escape(end_marker)}'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    return ""  # Return empty string if no match found

def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_core_text.py <input_file> <output_file> <markers_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    markers_file = sys.argv[3]

    # Read markers from the JSON file
    with open(markers_file, 'r') as f:
        markers = json.load(f)
    
    start_markers = markers.get('START_MARKERS', [])
    end_markers = markers.get('END_MARKERS', [])

    if not start_markers or not end_markers:
        print("Error: START_MARKERS and END_MARKERS must be provided in the markers file.")
        sys.exit(1)

    with open(input_file, 'r') as f:
        content = f.read()

    core_text = extract_core_text(content, start_markers, end_markers)

    with open(output_file, 'w') as f:
        f.write(core_text)

if __name__ == "__main__":
    main()