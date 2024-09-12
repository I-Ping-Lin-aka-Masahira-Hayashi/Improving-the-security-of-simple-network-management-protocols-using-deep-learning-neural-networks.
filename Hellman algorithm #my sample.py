#Below is my sample
import random

def diffie_hellman_key_exchange():
    # Prime number and generator
    p = 23
    g = 5

    # Private keys
    a = random.randint(1, p-1)
    b = random.randint(1, p-1)

    # Public keys
    A = pow(g, a, p)
    B = pow(g, b, p)

    # Shared secret
    secret_a = pow(B, a, p)
    secret_b = pow(A, b, p)

    if secret_a == secret_b:
        print(f'Successful key exchange, shared secret: {secret_a}')
    else:
        print('Key exchange failed')

diffie_hellman_key_exchange()
