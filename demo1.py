from tokenizers import Tokenizer
from pathlib import Path
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers import decoders

def smoke_test_tokenizer():
    """
    A simple smoke test to verify that the trained tokenizer
    can correctly encode and decode a sample text.
    """
    # --- 1. Load the tokenizer ---
    # Please ensure this path is correct
    tokenizer_path_str = "bpe_tokenizer_hf/openwebtext/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path_str)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    # --- 2. Define a sample text ---
    # This sample text includes spaces, punctuation, and newlines for a more comprehensive test.
    original_text = "Hello world! This is a test.\nNew line here."
    encoded = tokenizer.encode(original_text)
    encoded_ids = encoded.ids
    encoded_tokens = encoded.tokens
    print(f"Encoded Token IDs: {encoded_ids}")
    print(f"Encoded Tokens: {encoded_tokens}")
    decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
    print(f"Decoded text: '{decoded_text}'")

    if original_text == decoded_text:
        print("✅ Success: Decoded text matches the original text perfectly.")
    else:
        print("❌ Failure: Decoded text does not match the original text.")
        print(f"  Original: '{original_text}'")
        print(f"  Decoded: '{decoded_text}'")

    # Use an assertion for automated checking
    assert original_text == decoded_text, "Tokenizer encode/decode cycle failed!"
    
    # --- 3. Save the tokenizer (optional) ---
    output_dir = Path("/inspire/hdd/global_user/donglinkang-253108120084/standford-cs336/assignment1-basics/hf_tokenizer/openwebtext")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "tokenizer.json"
    tokenizer.save(str(model_file))

if __name__ == "__main__":
    smoke_test_tokenizer()