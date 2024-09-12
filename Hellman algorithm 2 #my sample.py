#Below is my sample
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import hashes

# 認証局の実装
ca = dh.generate_parameters(generator=2, key_size=2048)
ca_private_key = ca.generate_private_key()
ca_public_key = ca_private_key.public_key()

# Diffie-Hellman キー交換
client_private_key = ca.generate_private_key()
client_public_key = client_private_key.public_key()
server_private_key = ca.generate_private_key()
server_public_key = server_private_key.public_key()

shared_key = client_private_key.exchange(server_public_key)

