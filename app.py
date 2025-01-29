from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image
import torch
import time
from PIL import Image
from pymongo import MongoClient

# Define model paths and MongoDB URI
MODEL_PATH = "/app/models/paligemma2-3b-ft-docci-448"
EMOTION_MODEL_PATH = "/app/models/FaceScanPaliGemma_Emotion"
MODEL_PATH2 = "/app/models/paligemma-3b-pt-224"
MONGO_URI = "mongodb://mongodb:27017"
DB_NAME = "imageUploadDB"

# Load models and processors
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH)

emotion_analysis_model = PaliGemmaForConditionalGeneration.from_pretrained(EMOTION_MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True, device_map="auto")
emotion_analysis_processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH2, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MongoDB client and database
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
    This function retrieves all entries with the status 'uploaded' from all collections,
    sorted by upload date (oldest first).
    """
    pending_jobs = []

    # Iterate through all collections in the database
    for collection_name in db.list_collection_names():
        collection = db[collection_name]

        # Find all entries with the status 'uploaded' and sort by upload date
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
            backImagePath = oldest_job['entry']["backImagePath"]
            print("Analyzing image:", frontImagePath)

            # Update the status in the database to "generating"
            collection.update_one(
                {"_id": oldest_job['entry']['_id']}, 
                {"$set": {"status": "analyzing"}}
            )

            print("Loading")
            front_image = Image.open(frontImagePath)
            back_image = Image.open(backImagePath)

            prompts = [ '<image><bos> describe en\n',
                        '<image><bos> caption en\n',
            ]

            images = [front_image, back_image]

            model.to(device)
            model_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(torch.bfloat16).to(device)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=150, do_sample=False)
                generated_texts = generation[:, input_len:]
                decoded_texts = [processor.decode(text, skip_special_tokens=True) for text in generated_texts]

            front_image_description = decoded_texts[0]
            back_image_description = decoded_texts[1]

            emotion_analysis_model.to(device)
            input_text = "Answer en What is the emotion of the main person in the image? choose from: ‘neutral’, \t ‘happy’, \t ‘sad’ \t, ‘surprise’ \t ‘fear’ \t, ‘disgust’,\t ‘angry’ \n"
            inputs = emotion_analysis_processor(text=input_text, images=front_image, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
            inputs = inputs.to(dtype=emotion_analysis_model.dtype)

            with torch.no_grad():
                output = emotion_analysis_model.generate(**inputs, max_length=500)
                result=emotion_analysis_processor.decode(output[0], skip_special_tokens=True)[len(input_text):].strip()

            emotion = result

            collection.update_one(
                    {"_id": oldest_job['entry']['_id']}, 
                    {"$set": {"frontImageDescription": front_image_description}}
            )
            collection.update_one(
                    {"_id": oldest_job['entry']['_id']}, 
                    {"$set": {"backImageDescription": back_image_description}}
            )
            collection.update_one(
                    {"_id": oldest_job['entry']['_id']}, 
                    {"$set": {"emotion": emotion}}
            )
            
            # Update the status in the database to "prompt_created"
            collection.update_one(
                {"_id": oldest_job['entry']['_id']}, 
                {"$set": {"status": "analyzed"}}
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