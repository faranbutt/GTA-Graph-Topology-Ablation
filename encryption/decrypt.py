import sys
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

load_dotenv()

def decrypt_file_content(encrypted_file_path):
    private_key_pem = os.environ.get("SUBMISSION_PRIVATE_KEY")
    
    if not private_key_pem:
        raise ValueError("Error: 'SUBMISSION_PRIVATE_KEY' is missing from environment variables.")

    # If the key was stored as a single line with "\n" characters (common in .env files),
    # we must replace them with actual newline characters.
    private_key_pem = private_key_pem.replace('\\n', '\n')
    
    private_key_pem = private_key_pem.strip()

    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
    except Exception as e:
        print(f"DEBUG: Key starts with: {private_key_pem[:30]}...") 
        raise ValueError(f"Invalid Private Key format: {e}")

    try:
        with open(encrypted_file_path, "rb") as f:
            file_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {encrypted_file_path}")

    rsa_segment_size = 256 
    
    if len(file_content) < rsa_segment_size:
        raise ValueError("File is too short to contain a valid encrypted header.")

    encrypted_session_key = file_content[:rsa_segment_size]
    encrypted_data = file_content[rsa_segment_size:]

    try:
        session_key = private_key.decrypt(
            encrypted_session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    except Exception as e:
        raise ValueError(f"RSA Decryption failed. (Check if Public/Private keys match): {e}")

    try:
        cipher_suite = Fernet(session_key)
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return decrypted_data
    except Exception as e:
        raise ValueError(f"Data Decryption failed (Corrupted file?): {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decrypt.py <filename>")
    else:
        try:
            file_content = decrypt_file_content(sys.argv[1])
            new_file_name = sys.argv[1].replace(".enc", "")
            with open(new_file_name, "wb") as f:
                f.write(file_content)
            print(f"Decryption successful! Saved to '{new_file_name}'")
        except Exception as e:
            print(f"FAILED: {e}")