from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch
import time
from PIL import Image
from pymongo import MongoClient

MODEL_PATH = "/app/models/paligemma2-3b-ft-docci-448"
MONGO_URI = "mongodb://mongodb:27017"
DB_NAME = "imageUploadDB"

model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH)

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

try:
    # Test the connection by checking the server status
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print("MongoDB connection failed:", e)

def get_pending_jobs():
    """
    This function retrieves all entries with the status 'prompt_created' from all collections,
    sorted by creation date (oldest first).
    """
    pending_jobs = []

    # Iterate through all collections in the database
    for collection_name in db.list_collection_names():
        collection = db[collection_name]

        # Find all entries with the status 'prompt_created' and sort by creation date
        jobs = collection.find({"status": "uploaded"}).sort("uploadDate", 1)  # 1 for ascending order
        # Add the found jobs to the pending jobs list
        for job in jobs:
            pending_jobs.append({
                "collection": collection_name,
                "entry": job
            })

    # Sort the entire list of pending jobs by creation date
    pending_jobs.sort(key=lambda x: x["entry"]["uploadDate"])

    return pending_jobs


def analyze_images():
    jobs = get_pending_jobs()
    if not jobs:
        print("No pending jobs!")
        return

    while jobs:  # Continue processing until the job list is empty
        oldest_job = jobs.pop(0)  # Remove and get the first job from the list
        print(f"Processing job: {oldest_job}")
        
        collection_name = oldest_job['collection']
        collection = db[collection_name]

        # Update the status in the database to "create_image"
        collection.update_one(
            {"_id": oldest_job['entry']['_id']}, 
            {"$set": {"status": "create_image"}}
        )

        try:
            # Extract prompt
            frontImagePath = oldest_job['entry']["frontImagePath"]
            print("Analyzing image:", frontImagePath)

            # Update the status in the database to "generating"
            collection.update_one(
                {"_id": oldest_job['entry']['_id']}, 
                {"$set": {"status": "analyzing"}}
            )

            print("Loading")
            image = Image.open(frontImagePath)
            prompts = [ '<image><bos> answer en What is the gender of the person in the image? Male or female?\n',
                        '<image><bos> answer en What is the hair color, hair length and hairstyle of the person in the image?\n',
                        '<image><bos> answer en What is the eye color of the person in the image?\n',
                        '<image><bos> answer en What emotion is the person in the image strongly expressing?\n',
                        '<image><bos> answer en Summarize in maximum 6 words the clothing and its color.\n',
                        '<image><bos> answer en Summarize in 5 words: the background of the image?\n',
                        '<image><bos> answer en Is the person wearing glasses?\n',
            ]

            images = [image] * len(prompts)  # Assuming the same image for all questions

            model_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(torch.bfloat16).to(model.device)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
                generated_texts = generation[:, input_len:]
                decoded_texts = [processor.decode(text, skip_special_tokens=True) for text in generated_texts]

            text_prompt = f'A {decoded_texts[0]} with {decoded_texts[1]} hair, {decoded_texts[2]} eyes, strongly expressing {decoded_texts[3]} emotion, wearing {decoded_texts[4]}, in a setting with {decoded_texts[5]}, , icon emoji'
            print(text_prompt)
            collection.update_one(
                    {"_id": oldest_job['entry']['_id']}, 
                    {"$set": {"prompt": text_prompt}}
            )
            
            # Update the status in the database to "prompt_created"
            collection.update_one(
                {"_id": oldest_job['entry']['_id']}, 
                {"$set": {"status": "prompt_created"}}
            )

        except Exception as e:
            print(f"Error analyzing image for job {oldest_job['entry']['_id']}: {e}")
        
# Loop, 15s intervals
while True:
    try:
        analyze_images()
    except Exception as e:
        print("Error: ", e)
    print("Warte 15 Sekunden...")
    time.sleep(15)