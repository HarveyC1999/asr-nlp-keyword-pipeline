#!/usr/bin/env python
# coding: utf-8
import cx_Oracle
import pandas as pd 
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
import datetime
import warnings
warnings.filterwarnings("ignore")



##DB login
def get_key(key_file):
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key

def decrypt_data(encrypt_msg):
    private_key = get_key(r'rsa_private_key.pem')
    cipher = PKCS1_cipher.new(private_key)
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg), 0)
    return back_text.decode('utf-8')


import re
import time
import warnings
warnings.filterwarnings("ignore")


def replace_text(text):
    cur.execute("""
    select incorrect_word,correct_word 
    from stt_replace_word
    where status = 'Y'""")
    corrections =dict(cur.fetchall())
    for incorrect, correct in corrections.items():
        if incorrect in text:
            text = text.replace(f'{incorrect}',f'{correct}')
    return text

def format_time(seconds):
    if seconds >= 3600:
        return time.strftime('%H:%M:%S', time.gmtime(seconds))
    else:
        return time.strftime('%M:%S', time.gmtime(seconds))

