
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

public_key = private_key.public_key()
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)


if __name__ == "__main__":
    import os
    if os.path.exists("private_key.pem") or os.path.exists("public_key.pem"):
        print("Warning: Key files already exist. Overwriting them will cause loss of access to encrypted files.")
        print("If this is intentional, please backup and delete existing 'private_key.pem' and 'public_key.pem' files before running this script again.")
        exit(1)
        
    with open("private_key.pem", "wb") as f:
        f.write(pem_private)
    with open("public_key.pem", "wb") as f:
        f.write(pem_public)

    print("Keys generated!")