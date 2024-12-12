import sys
import os
import random
import string
import ctypes
from ctypes import wintypes

import pyperclip

advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)


USERNAME = bytes(os.environ.get('USERNAME'), 'utf-8')
STRINGS = string.ascii_letters + string.punctuation + string.digits

CRED_TYPE_GENERIC = 1
CRED_PERSIST_ENTERPRISE = 3


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD)]


class CREDENTIAL_ATTRIBUTEA(ctypes.Structure):
    _fields_ = [("Keyword", wintypes.LPSTR),
                ("Flags", wintypes.DWORD),
                ("ValueSize", wintypes.DWORD),
                ("Value", wintypes.LPVOID)]


class CREDENTIALA(ctypes.Structure):
    _fields_ = [("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPSTR),
                ("Comment", wintypes.LPSTR),
                ("LastWritten", FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", wintypes.LPBYTE),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.POINTER(CREDENTIAL_ATTRIBUTEA)),
                ("TargetAlias", wintypes.LPSTR),
                ("UserName", wintypes.LPSTR)]


CredReadA = advapi32.CredReadA
CredFree = advapi32.CredFree
CredWriteA = advapi32.CredWriteA


def create_passwd(length: int = 20):
    return ''.join(random.choices(STRINGS, k=length))


def read(target):
    cred_ptr = ctypes.POINTER(CREDENTIALA)()

    success = CredReadA(
        bytes(target, 'utf-8'), CRED_TYPE_GENERIC, 0, ctypes.byref(cred_ptr)
    )
    if not success:
        print('Not found.')
        return 1

    cred = cred_ptr.contents
    blob = bytes(cred.CredentialBlob[:cred.CredentialBlobSize]).decode()

    if not blob:
        print('Empty credential.')
        return 0

    pyperclip.copy(blob)

    CredFree(cred_ptr)

    print('The password was written on the clipboard.')
    return success


def write(target):
    passwd = ctypes.create_string_buffer(bytes(create_passwd(), 'utf-8'))

    cred = CREDENTIALA()
    cred.Flags = 0
    cred.Type = CRED_TYPE_GENERIC
    cred.TargetName = bytes(target, 'utf-8')
    cred.CredentialBlobSize = len(passwd)
    cred.CredentialBlob = ctypes.cast(passwd, wintypes.LPBYTE)
    cred.Persist = CRED_PERSIST_ENTERPRISE
    cred.UserName = USERNAME

    success = CredWriteA(ctypes.byref(cred), 0)
    if success:
        print('Credential written.')
    else:
        print('Error writting credential.')

    return success


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '-r':
        sys.exit(read(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == '-w':
        sys.exit(write(sys.argv[2]))
    else:
        print(
            'usage: cred {{-r | -w} <target>}'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
