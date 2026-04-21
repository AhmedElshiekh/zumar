import os
import urllib.request

def download_tokenizer():
    # نصعد مستويين للوصول للجذر ثم مجلد models
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    target_dir = os.path.join(BASE_DIR, "models", "zumar-v1")
    
    os.makedirs(target_dir, exist_ok=True)
    
    url = "https://huggingface.co/openai-community/gpt2/raw/main/tokenizer.json"
    target_file = os.path.join(target_dir, "tokenizer.json")
    
    print(f"📥 Downloading tokenizer to: {target_file}")
    
    try:
        urllib.request.urlretrieve(url, target_file)
        print("✅ Tokenizer ready for Sovereign Inference.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_tokenizer()
