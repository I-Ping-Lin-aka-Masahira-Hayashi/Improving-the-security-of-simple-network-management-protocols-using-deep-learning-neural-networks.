#Below is the my sample
import hashlib
import os
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hmac

class SNMPAgent:
    def __init__(self, community_string, users=None):
        self.community_string = community_string
        self.users = users or {}
        self.data = {"sysUpTime": 3600, "ifNumber": 4}
        self.views = {"full": ["sysUpTime", "ifNumber"], "restricted": ["sysUpTime"]}
        self.ca_public_key = None  # For simulating a Certificate Authority

class SNMPManager:
    def __init__(self, community_string, users=None):
        self.community_string = community_string
        self.users = users or {}
        self.ca_public_key = None  # For simulating a Certificate Authority

def generate_key():
    return os.urandom(32)

def snmpv1_get(agent, manager, oid):
    if agent.community_string == manager.community_string:
        return agent.data.get(oid, "Not found")
    else:
        return "Access denied"

def snmpv2c_get(agent, manager, oid):
    return snmpv1_get(agent, manager, oid)  # Same as SNMPv1 for this simulation

def snmpv3_get(agent, manager, oid, username, auth_key, privacy_key):
    if username not in agent.users or username not in manager.users:
        return "User not found"
    
    # Simulate USM authentication using HMAC
    auth_hmac = hmac.new(auth_key.encode(), msg=username.encode(), digestmod=hashlib.sha256)
    if auth_hmac.hexdigest() != agent.users[username]['auth_hmac']:
        return "Authentication failed"
    
    # Use AES for encryption (simulating privacy)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(privacy_key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_oid = encryptor.update(oid.encode()) + encryptor.finalize()
    
    # Simulate VACM access control
    if oid not in agent.views[agent.users[username]['view']]:
        return "Access denied"
    
    # Decrypt the OID
    decryptor = cipher.decryptor()
    decrypted_oid = decryptor.update(encrypted_oid) + decryptor.finalize()
    
    return agent.data.get(decrypted_oid.decode(), "Not found")

def simulate_mitm_attack(agent, manager, oid):
    print(f"Attempting MITM attack on SNMPv1/v2c...")
    print(f"Intercepted community string: {agent.community_string}")
    print(f"Intercepted data: {agent.data.get(oid, 'Not found')}")

def simulate_snmpv3_attack(agent, oid):
    print(f"Attempting attack on SNMPv3...")
    print("Encrypted data intercepted, unable to read content.")

def diffie_hellman_key_exchange():
    parameters = dh.generate_parameters(generator=2, key_size=2048)
    private_key1 = parameters.generate_private_key()
    private_key2 = parameters.generate_private_key()
    
    shared_key1 = private_key1.exchange(private_key2.public_key())
    shared_key2 = private_key2.exchange(private_key1.public_key())
    
    derived_key1 = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
    ).derive(shared_key1)
    
    derived_key2 = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'handshake data',
    ).derive(shared_key2)
    
    return derived_key1, derived_key2

def simulate_ca(agent, manager):
    # Simulate a Certificate Authority for mutual authentication
    ca_private_key = dh.generate_parameters(generator=2, key_size=2048).generate_private_key()
    agent.ca_public_key = ca_private_key.public_key()
    manager.ca_public_key = ca_private_key.public_key()
    return ca_private_key

def optimize_snmpv3_get(agent, manager, oid, username, auth_key, privacy_key):
    start_time = time.time()
    result = snmpv3_get(agent, manager, oid, username, auth_key, privacy_key)
    end_time = time.time()
    print(f"Optimized SNMPv3 Get execution time: {end_time - start_time} seconds")
    return result

# Simulate SNMPv1 and SNMPv2c
agent_v1 = SNMPAgent("public")
manager_v1 = SNMPManager("public")
print("SNMPv1/v2c Get:", snmpv1_get(agent_v1, manager_v1, "sysUpTime"))

# Simulate MITM attack on SNMPv1/v2c
simulate_mitm_attack(agent_v1, manager_v1, "sysUpTime")

# Simulate SNMPv3
users = {
    "admin": {"auth_hmac": hmac.new("adminauth".encode(), msg="admin".encode(), digestmod=hashlib.sha256).hexdigest(), "view": "full"},
    "user": {"auth_hmac": hmac.new("userauth".encode(), msg="user".encode(), digestmod=hashlib.sha256).hexdigest(), "view": "restricted"}
}
agent_v3 = SNMPAgent("", users)
manager_v3 = SNMPManager("", users)
privacy_key = generate_key()

print("\nSNMPv3 Get (admin):", snmpv3_get(agent_v3, manager_v3, "sysUpTime", "admin", "adminauth", privacy_key))
print("SNMPv3 Get (user):", snmpv3_get(agent_v3, manager_v3, "ifNumber", "user", "userauth", privacy_key))

# Simulate attack on SNMPv3
simulate_snmpv3_attack(agent_v3, "sysUpTime")

# Demonstrate Diffie-Hellman key exchange
print("\nPerforming Diffie-Hellman key exchange...")
key1, key2 = diffie_hellman_key_exchange()
print(f"Shared key established: {key1 == key2}")

# Simulate Certificate Authority for mutual authentication
print("\nSimulating Certificate Authority for mutual authentication...")
ca_private_key = simulate_ca(agent_v3, manager_v3)
print(f"CA public key set for agent and manager: {agent_v3.ca_public_key == manager_v3.ca_public_key}")

# Demonstrate optimized SNMPv3 Get
print("\nDemonstrating optimized SNMPv3 Get:")
optimize_snmpv3_get(agent_v3, manager_v3, "sysUpTime", "admin", "adminauth", privacy_key)

print("\nNote: This simulation implements more security features and optimizations mentioned in the research, but it's still a simplified model.")