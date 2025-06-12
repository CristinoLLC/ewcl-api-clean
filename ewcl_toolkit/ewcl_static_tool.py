import os
import requests
import json

def call_ewcl_api(file_path: str, reverse: bool = False):
    """
    Sends a POST request to the EWCL API and saves the response as a JSON file.

    Args:
        file_path (str): Path to the .pdb file to upload.
        reverse (bool): Whether to call the reverse endpoint.

    Returns:
        dict: The full JSON response from the API.
    """
    # Determine the endpoint and mode
    url = "https://ewcl-platform.onrender.com/analyze-rev" if reverse else "https://ewcl-platform.onrender.com/analyze"
    mode = "reverse" if reverse else "normal"

    # Prepare output directory and file name
    output_dir = "./ewcl_api_outputs"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, f"{base_name}.{mode}.json")

    # Check if the result already exists
    if os.path.exists(output_file):
        print(f"✅ Skipping {file_path} ({mode}): Result already exists at {output_file}")
        with open(output_file, "r") as f:
            return json.load(f)

    # Upload the file and make the API call
    with open(file_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                result = response.json()
                result["mode"] = mode  # Add mode to the result

                # Save the result to a JSON file
                with open(output_file, "w") as out_f:
                    json.dump(result, out_f, indent=4)

                print(f"✅ Saved result for {file_path} ({mode}) to {output_file}")
                return result
            else:
                print(f"❌ Error for {file_path} ({mode}): {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Exception for {file_path} ({mode}): {str(e)}")
            return None