from sage.all import GF
import os
from Crypto.Cipher import AES

x = GF(2)["x"].gen()
gf2e = GF(2 ** 128, name="y", modulus=x ** 128 + x ** 7 + x ** 2 + x + 1)


# Converts an integer to a gf2e element, little endian.
def _to_gf2e(n):
    return gf2e([(n >> i) & 1 for i in range(127, -1, -1)])


# Converts a gf2e element to an integer, little endian.
def _from_gf2e(p):
    n = p.integer_representation()
    ans = 0
    for i in range(128):
        ans <<= 1
        ans |= ((n >> i) & 1)

    return ans


# Calculates the GHASH polynomial.
def _ghash(h, a, c):
    la = len(a)
    lc = len(c)
    p = gf2e(0)
    for i in range(la // 16):
        p += _to_gf2e(int.from_bytes(a[16 * i:16 * (i + 1)], byteorder="big"))
        p *= h

    if la % 16 != 0:
        p += _to_gf2e(int.from_bytes(a[-(la % 16):] + bytes(16 - la % 16), byteorder="big"))
        p *= h

    for i in range(lc // 16):
        p += _to_gf2e(int.from_bytes(c[16 * i:16 * (i + 1)], byteorder="big"))
        p *= h

    if lc % 16 != 0:
        p += _to_gf2e(int.from_bytes(c[-(lc % 16):] + bytes(16 - lc % 16), byteorder="big"))
        p *= h

    p += _to_gf2e(((8 * la) << 64) | (8 * lc))
    p *= h
    return p


def recover_possible_auth_keys(a1, c1, t1, a2, c2, t2):
    """
    Recovers possible authentication keys from two messages encrypted with the same authentication key.
    More information: Joux A., "Authentication Failures in NIST version of GCM"
    :param a1: the associated data of the first message (bytes)
    :param c1: the ciphertext of the first message (bytes)
    :param t1: the authentication tag of the first message (bytes)
    :param a2: the associated data of the second message (bytes)
    :param c2: the ciphertext of the second message (bytes)
    :param t2: the authentication tag of the second message (bytes)
    :return: a generator generating possible authentication keys (gf2e element)
    """
    h = gf2e["h"].gen()
    p1 = _ghash(h, a1, c1) + _to_gf2e(int.from_bytes(t1, byteorder="big"))
    p2 = _ghash(h, a2, c2) + _to_gf2e(int.from_bytes(t2, byteorder="big"))
    for h, _ in (p1 + p2).roots():
        yield h


def forge_tag(h, a, c, t, target_a, target_c):
    """
    Forges an authentication tag for a target message given a message with a known tag.
    This method is best used with the authentication keys generated by the recover_possible_auth_keys method.
    More information: Joux A., "Authentication Failures in NIST version of GCM"
    :param h: the authentication key to use (gf2e element)
    :param a: the associated data of the message with the known tag (bytes)
    :param c: the ciphertext of the message with the known tag (bytes)
    :param t: the known authentication tag (bytes)
    :param target_a: the target associated data (bytes)
    :param target_c: the target ciphertext (bytes)
    :return: the forged authentication tag (bytes)
    """
    ghash = _from_gf2e(_ghash(h, a, c))
    target_ghash = _from_gf2e(_ghash(h, target_a, target_c))
    return (ghash ^ int.from_bytes(t, byteorder="big") ^ target_ghash).to_bytes(16, byteorder="big")


secret = os.urandom(16)
key  = b'\xb2\x86T%D\x04\xd8}w\xaf\xcb\xdbX\x9c\xb8i'
iv = b'X\x1b%H\xdf\xd2\xdb\x97\xc4\x01A\xbc'


def get_mac(message):
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    _, mac = cipher.encrypt_and_digest(message)
    return mac

def bxor(a, b):
    assert len(a) == len(b)
    return bytes([x ^ y for x, y in zip(a, b)])

c1 = get_mac(b"\x00" * 32 + secret + b"\x00" * 16)
c2 = get_mac(b"\x00" * 32 + secret + b"\x01" * 16)
c3 = get_mac(b"\x00" * 16 + secret + b"\x00" * 32)

c = _to_gf2e(int.from_bytes(bxor(c1, c2), 'big')) / _to_gf2e(int.from_bytes(b"\x01" * 16, 'big'))
h = c.nth_root(2)

d = _to_gf2e(int.from_bytes(bxor(c1, c3), 'big'))
s = d/(h**4+h**3)
guess_secret = _from_gf2e(s).to_bytes(16, 'big')
assert guess_secret == secret

from pwn import process, remote
# io = process(["python3", "server.py"])
# nc chall.lac.tf 32222
io = remote("chall.lac.tf", 32222)

def get_mac_p(l:bytes, r:bytes):
    io.sendline(b"1")
    io.sendline(l.hex().encode())
    io.recvuntil(b"input > ")
    io.sendline(r.hex().encode())
    io.recvuntil(b"input > ")
    mac = io.recvline().strip().decode()
    print(mac)
    return bytes.fromhex(mac)

c1 = get_mac_p(b"\x00" * 32, b"\x00" * 16)
c2 = get_mac_p(b"\x00" * 32, b"\x01" * 16)
c3 = get_mac_p(b"\x00" * 16, b"\x00" * 32)

c = _to_gf2e(int.from_bytes(bxor(c1, c2), 'big')) / _to_gf2e(int.from_bytes(b"\x01" * 16, 'big'))
h = c.nth_root(2)

d = _to_gf2e(int.from_bytes(bxor(c1, c3), 'big'))
s = d/(h**4+h**3)
guess_secret = _from_gf2e(s).to_bytes(16, 'big')


io.sendline(b"2")
io.sendline(guess_secret.hex().encode())
io.interactive()

