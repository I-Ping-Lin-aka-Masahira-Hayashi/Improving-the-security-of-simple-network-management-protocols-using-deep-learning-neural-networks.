#To solve the SNMP security exchange key problem, I would prefer to improve it using the Diffie-Hellman algorithm.

import random 
    
def SNMPDiffie():
    p=23
    g=7

    a=random.randint(1,p-1)
    b=random.randint(1,p-1)

    A=pow(g,a,p)
    B=pow(g,b,p)

    secret_a=pow(B,a,p)
    secret_b=pow(A,b,p)

    if secret_a==secret_b:
        print(f'Diffie-Hellman algorithm for SNMP security is improved by R. Mohsin et al. in 2019. The shared secret is: {secret_a} ')

SNMPDiffie()

