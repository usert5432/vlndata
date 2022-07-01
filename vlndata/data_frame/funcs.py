import gzip
import lzma
import os
import sys

from io import BufferedReader, BytesIO
from typing import Any, Union

if sys.version_info[:2] >= (3,8):
    from multiprocessing.shared_memory import SharedMemory
else:
    SharedMemory = Any

def get_f_size(f : BufferedReader) -> int:
    f.seek(0, 2)
    return f.tell()

def read_into_shmem(f : BufferedReader) -> SharedMemory:
    size   = get_f_size(f)
    result = SharedMemory(create = True, size = size)

    f.seek(0, 0)
    f.readinto(result.buf)

    return result

def load_file_into_shmem(path : Union[str, BufferedReader]) -> SharedMemory:
    if isinstance(path, str):
        _basename, ext = os.path.splitext(path)

        if ext in [ '.xz', '.lz', '.lzip' ]:
            f = lzma.open(path, mode = 'rb')
        elif ext in [ '.gz', '.gzip' ]:
            f = gzip.open(path, mode = 'rb')
        else:
            f = open(path, mode = 'rb')
    else:
        f = path

    result = read_into_shmem(f)
    f.close()

    return result

def make_buffer_from_shmem(shmem : SharedMemory) -> BytesIO:
    return BytesIO(shmem.buf)

