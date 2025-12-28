import google.generativeai as genai

# ASK FOR KEY
key = input("Paste your API Key here: ").strip()
genai.configure(api_key=key)

print("\n🔍 Checking available models for this key...\n")

try:
    count = 0
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
            count += 1
    
    if count == 0:
        print("❌ No text generation models found! Your API key might be restricted.")
    else:
        print(f"\n🎉 Success! Found {count} models.")

except Exception as e:
    print(f"\n❌ ERROR: Your API Key is invalid or there is a connection issue.")
    print(f"Error details: {e}")