#Below is my sample
import hashlib
import os
from cryptography.fernet import Fernet

class SNMPAgent:
    def __init__(self, community_string):
        self.community_string = community_string
        self.data = {"sysUpTime": 3600, "ifNumber": 4}

class SNMPManager:
    def __init__(self, community_string):
        self.community_string = community_string

def generate_key():
    return Fernet.generate_key()

def snmpv1_get(agent, manager, oid):
    if agent.community_string == manager.community_string:
        return agent.data.get(oid, "Not found")
    else:
        return "Access denied"

def snmpv3_get(agent, manager, oid, username, auth_key, privacy_key):
    # Simulate USM authentication
    auth_hash = hashlib.sha256(f"{username}{auth_key}".encode()).hexdigest()
    
    # Use the provided key for Fernet
    fernet = Fernet(privacy_key)
    encrypted_oid = fernet.encrypt(oid.encode())
    
    # Simulate VACM access control
    if auth_hash == hashlib.sha256(f"{username}{agent.community_string}".encode()).hexdigest():
        decrypted_oid = fernet.decrypt(encrypted_oid).decode()
        return agent.data.get(decrypted_oid, "Not found")
    else:
        return "Access denied"

def simulate_snmp_attack(agent, oid):
    # Simulate a basic sniffing attack on SNMPv1/v2
    print(f"Intercepted SNMP request for OID: {oid}")
    print(f"Intercepted community string: {agent.community_string}")
    print(f"Intercepted data: {agent.data.get(oid, 'Not found')}")

# Simulate SNMPv1 communication
agent_v1 = SNMPAgent("public")
manager_v1 = SNMPManager("public")
print("SNMPv1 Get:", snmpv1_get(agent_v1, manager_v1, "sysUpTime"))

# Simulate SNMPv1 attack
print("\nSimulating SNMPv1 attack:")
simulate_snmp_attack(agent_v1, "sysUpTime")

# Simulate SNMPv3 communication
agent_v3 = SNMPAgent("secretstring")
manager_v3 = SNMPManager("secretstring")
username = "admin"
auth_key = "authpassword"
privacy_key = generate_key()  # Generate a proper Fernet key
print("\nSNMPv3 Get:", snmpv3_get(agent_v3, manager_v3, "sysUpTime", username, auth_key, privacy_key))

# Attempt SNMPv3 attack (which will fail due to encryption and authentication)
print("\nAttempting SNMPv3 attack:")
simulate_snmp_attack(agent_v3, "sysUpTime")