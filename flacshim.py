#!/usr/bin/env python
# flacshim -- store separate metadata blocks for FLAC files
# Copyright (c) 2015 Ian Weller <ianweller@buttslol.net>
#
# Based on the loopback.py example provided with fusepy
# Copyright (c) 2012 Terence Honles <terence@honles.com>
# Copyright (c) 2008 Giorgos Verigakis <verigak@gmail.com>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""
flacshim is a FUSE filesystem to transparently store metadata blocks for FLAC
files in a separate area while keeping original FLAC files untouched.

Operation is as follows:
- for non-FLAC files, operations are passed through to the "data" directory
  within the base
- for FLAC files (i.e. starting with "fLaC"), all standard metadata blocks
  (except SEEKTABLE and PADDING) are stored in the "data" directory. The first
  4 bytes of the file will be "FlAc" to differentiate from an actual FLAC file.
  flacshim also (poorly) assumes anything starting with "fLaC" is a valid FLAC.

A "flacmd5" directory will exist in the base directory, which contains files
(possibly links to files) where the name is the FLAC MD5 signature (the MD5 sum
of the raw audio frames). When reading a partial FLAC ("FlAc"), flacshim will
find the original FLAC under flacmd5, and read out:
- the STREAMINFO block from the partial file,
- the SEEKTABLE from the original file,
- all other blocks stored in the partial file,
- a 4096-bit PADDING block,
- the audio frames from the original file.

Non-defined blocks are ignored by flacshim.

TODO: deduplicate PICTURE blocks?
"""

from __future__ import with_statement

import errno
import fuse
import io
import itertools
import os
import struct
import sys
import threading


def regular_flac(path):
    """
    Return True if the file starts with "fLaC" and is thus a FLAC file.
    """
    with open(path, 'rb') as f:
        return f.read(4) == 'fLaC'


def partial_flac(path):
    """
    Return True if the file starts with "FlAc" (inverted case of "fLaC", the
    bytes that start a normal FLAC file).
    """
    try:
        with open(path, 'rb') as f:
            return f.read(4) == 'FlAc'
    except:
        return False


def parse_metadata_block_header(data):
    data = struct.unpack('>I', data)[0]
    return {'last': bool(data >> 31),
            'block_type': (data >> 24) & 0x7f,
            'length': data & 0xffffff}


def read_metadata_block_headers(f, callback):
    """
    Calls callback(f, pos, header) where pos is the position where the metadata
    block header starts, and header is a dict with keys ('last', 'block_type',
    'length'). The position of f is after the metadata block header. Assumes
    the original position of the file is at 4 (after the fLaC).
    """
    while True:
        pos = f.tell()
        header = parse_metadata_block_header(f.read(4))
        callback(f, pos, header)

        f.seek(header['length'], os.SEEK_CUR)
        if header['last']:
            break


class ReassembledFlac(object):

    def __init__(self, path, flacmd5_base):
        self.path = path
        self.flacmd5_base = flacmd5_base
        self.pos = 0
        flacmd5 = None

        # these variables consist of (file, offset, size) tuples for each block
        self.streaminfo = None  # from partial
        self.seektable = None  # from original
        self.others = []  # from partial
        self.frames = None  # from original
        self.flacmd5 = None

        self.partial = open(self.path, 'rb')
        assert self.partial.read(4) == 'FlAc'

        def partial_header_callback(f, pos, header):
            if self.streaminfo is None and header['block_type'] == 0:
                self.streaminfo = (self.partial, pos, 4 + header['length'])
                f.seek(18, os.SEEK_CUR)
                self.flacmd5 = f.read(16).encode('hex')
                f.seek(-34, os.SEEK_CUR)
            elif header['block_type'] in (2, 4, 5, 6):
                self.others.append((self.partial, pos, 4 + header['length']))
        read_metadata_block_headers(self.partial, partial_header_callback)

        assert self.streaminfo is not None
        assert self.flacmd5 is not None

        self.original = open(os.path.join(self.flacmd5_base, self.flacmd5), 'rb')
        assert self.original.read(4) == 'fLaC'

        def original_header_callback(f, pos, header):
            if self.seektable is None and header['block_type'] == 3:
                self.seektable = (self.original, pos, 4 + header['length'])
        read_metadata_block_headers(self.original, original_header_callback)

        assert self.seektable is not None
        pos = self.original.tell()
        self.original.seek(0, os.SEEK_END)
        self.frames = (self.original, pos, self.original.tell() - pos)

        # self.chunks: (dis_start, length, file, file_start)
        self.chunks = [(0, 4, io.BytesIO('fLaC'), 0)]
        for chunk in itertools.chain((self.streaminfo, self.seektable),
                                     self.others):
            dis_start = sum(self.chunks[-1][0:2])
            # ensure the frame header doesn't have the last bit on
            chunk[0].seek(chunk[1])
            frame_header = struct.unpack('>I', chunk[0].read(4))[0]
            frame_header &= 0x7fffffff
            self.chunks.append((dis_start, 4,
                                io.BytesIO(struct.pack('>I', frame_header)),
                                0))
            dis_start = sum(self.chunks[-1][0:2])
            self.chunks.append((dis_start, chunk[2] - 4,
                                chunk[0], chunk[1] + 4))
        for chunk in ((io.BytesIO('\x81\x00\x10\x00' +
                       ('\x00' * 4096)), 0, 4100), self.frames):
            dis_start = sum(self.chunks[-1][0:2])
            self.chunks.append((dis_start, chunk[2], chunk[0], chunk[1]))
        self.is_open = True

    def close(self):
        self.is_open = False
        self.partial.close()
        self.original.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def size(self):
        return sum(self.chunks[-1][0:2])

    def seek(self, offset, whence):
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        elif whence == os.SEEK_END:
            self.pos = self.size - offset

    def read(self, size=-1):
        value = b''
        for dis_start, length, f, f_start in self.chunks:
            if self.pos >= dis_start + length:
                continue
            f.seek((self.pos - dis_start) + f_start)
            chunk_remaining = length - (self.pos - dis_start)
            if size < 0 or size > chunk_remaining:
                value += f.read(chunk_remaining)
                self.pos += chunk_remaining
                size -= chunk_remaining
            else:
                self.pos += size
                value += f.read(size)
                break
        return value


class flacshim(fuse.LoggingMixIn, fuse.Operations):

    def __init__(self, base):
        self.base = os.path.realpath(base)
        self.root = os.path.join(self.base, 'data')
        self.flacmd5_base = os.path.join(self.base, 'flacmd5')
        self.rwlock = threading.Lock()

        self.reassembled = {}
        self.flags = {}

    def __call__(self, op, path, *args):
        """
        Prepends self.root to path in every call.

        Excellent for a purely passthrough filesystem, but can be annoying --
        helper functions to analyze files in self.root are outside this class
        for this reason.
        """
        return super(flacshim, self).__call__(op, self.root + path, *args)

    def access(self, path, mode):
        if not os.access(path, mode):
            raise fuse.FuseOSError(errno.EACCES)

    chmod = os.chmod
    chown = os.chown

    def create(self, path, mode):
        return os.open(path, os.O_WRONLY | os.O_CREAT, mode)

    def flush(self, path, fh):
        return os.fsync(fh)

    def fsync(self, path, datasync, fh):
        return os.fsync(fh)

    def getattr(self, path, fh=None):
        st = os.lstat(path)
        ret = dict((key, getattr(st, key)) for key in ('st_atime', 'st_ctime',
            'st_gid', 'st_mode', 'st_mtime', 'st_nlink', 'st_size', 'st_uid'))
        if partial_flac(path):
            with ReassembledFlac(path, self.flacmd5_base) as rf:
                ret['st_size'] = rf.size
        return ret

    getxattr = None

    def link(self, target, source):
        return os.link(source, target)

    listxattr = None
    mkdir = os.mkdir
    mknod = os.mknod

    def open(self, path, flags):
        if partial_flac(path):
            fh = os.open('/dev/null', flags)
            self.reassembled[fh] = ReassembledFlac(path, self.flacmd5_base)
            self.flags[fh] = flags
            return fh
        else:
            return os.open(path, flags)

    def read(self, path, size, offset, fh):
        with self.rwlock:
            if fh in self.reassembled:
                flac = self.reassembled[fh]
                flac.seek(offset, os.SEEK_SET)
                return flac.read(size)
            else:
                os.lseek(fh, offset, 0)
                return os.read(fh, size)

    def readdir(self, path, fh):
        return ['.', '..'] + os.listdir(path)

    readlink = os.readlink

    def release(self, path, fh):
        # FIXME copying a file out results in:
        # cp: failed to close 'music/01 Powerup.flac': Invalid argument
        if fh in self.reassembled:
            self.reassembled[fh].close()
            del self.reassembled[fh]
            del self.flags[fh]
        elif regular_flac(path):
            with self.rwlock:
                pass
                # disassemble the FLAC
                raise NotImplementedError()
        return os.close(fh)

    def rename(self, old, new):
        return os.rename(old, self.root + new)

    rmdir = os.rmdir

    def statfs(self, path):
        stv = os.statvfs(path)
        return dict((key, getattr(stv, key)) for key in ('f_bavail', 'f_bfree',
            'f_blocks', 'f_bsize', 'f_favail', 'f_ffree', 'f_files', 'f_flag',
            'f_frsize', 'f_namemax'))

    def symlink(self, target, source):
        return os.symlink(source, target)

    def truncate(self, path, length, fh=None):
        with open(path, 'r+') as f:
            f.truncate(length)

    unlink = os.unlink
    utimens = os.utime

    def write(self, path, data, offset, fh):
        with self.rwlock:
            # If it's a partial FLAC, write the reassembled FLAC out to the
            # file, close it, and turn the fh into one representing that file.
            # We turn it back into a disassembled FLAC in release()
            if fh in self.reassembled:
                self.reassembled[fh].seek(0, 0)
                with open(path, 'wb') as f:
                    f.write(self.reassembled[fh].read())
                self.reassembled[fh].close()
                new_fh = os.open(path, self.flags[fh])
                os.dup2(new_fh, fh)
                del self.reassembled[fh]
                del self.flags[fh]
                fh = new_fh
            os.lseek(fh, offset, 0)
            return os.write(fh, data)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: %s <base> <mountpoint>' % sys.argv[0])
        sys.exit(1)

    fuse.FUSE(flacshim(sys.argv[1]), sys.argv[2], foreground=True)
