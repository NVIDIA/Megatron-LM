from __future__ import print_function

import errno
import logging
import os
import re
import socket
import sys

from pdb import Pdb

import torch

__author__ = "LEI WANG (yiak.wy@gmail.com)"
__date__ = "15-July-2024"
__doc__ = "Adapted from github.com/tamentis/rpdb to debug multi-nodes Megatron-lm efficiently"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

COLOR_OFF='\033[0m'
BOLD='\033[1m'


def log(message, stderr=sys.__stderr__):
    _logger.info(message)
    print(message, file=stderr)
    stderr.flush()

class FileWrapper(object):

    def __init__(self, conn, io):
        self.conn = conn
        self.stream = conn.makefile('rw')
        self._io = io
        self._pat = re.compile("\r?\n")

    # a proxy of original file object
    def __getattr__(self, attr):
        if hasattr(self.stream, attr):
            attr = getattr(self.stream, attr)
        elif hasattr(self._io, attr):
            attr = getattr(self._io, attr)
        else:
            raise AttributeError(f"Expected attribute <{attr}>, but not found in {__class__.__name__}.")
        
        return attr
    
    def write(self, data, pat=None):
        pat = pat or self._pat
        data = pat.sub("\r\n", data)
        self._send(data)

    def writelines(self, lines, pat=None):
        pat = pat or self._pat
        for line in lines:
            self.write(line, pat)

    def _send(self, data):
        if hasattr(self.stream, 'encoding'):
            data = data.encode(self.stream.encoding)

        self.conn.sendall(data)


class RemotePdb(Pdb):
    """
    This code servers as telnet service. It will not block the codes until a client has connected.
    """

    def __init__(self, host, port):
        self.rank = torch.distributed.get_rank()

        # open up a socket
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.listen_socket.bind((host, port))

        if host == '0.0.0.0' or host == 'localhost':
            # single node debug
            ip = socket.gethostbyname(socket.gethostname())
        else:
            ip = host

        log(f"[Rank {self.rank}] [remote pdb endpoint] use cmd {BOLD}<rlwrap socat - tcp:{ip}:{self.listen_socket.getsockname()[1]}{COLOR_OFF}> to connect remote pdb")
        
        self.listen_socket.listen(1)
        (conn, address) = self.listen_socket.accept()

        log(f"[Rank {self.rank}] remote pdb accepts a connection from {address}")

        self.stdin_handle = FileWrapper(conn, sys.stdin)
        
        Pdb.__init__(self, completekey='tab', stdin=self.stdin_handle, stdout=self.stdin_handle)
        
        self.backup()

        RemotePdb.active_instance = self
        pass

    def backup(self):
        self.old_streams = []
        assert self.stdin_handle is not None

        for name in ('stdin', 'stdout'):
            self.old_streams.append((name, getattr(sys, name)))
            setattr(sys, name, self.stdin_handle)
        pass

    def restore(self):
        log(f"[Rank {self.rank}] recovering streams : ...")
        for name, stream in self.old_streams:
            setattr(sys, name, stream)
        log(f"[Rank {self.rank}] streams recovered.")
        pass

    # close stdin_handle
    def shutdown(self):
        self.restore()

        self.stdin_handle.close()

        self.listen_socket.shutdown(socket.SHUT_RDWR)
        self.listen_socket.close()

        RemotePdb.active_instance = None
        pass

    def do_quit(self, arg):
        try:
            return Pdb.do_quit(self, arg)
        finally:
            self.shutdown()

    do_q = do_exit = do_quit

    def do_continue(self, arg):
        try:
            return Pdb.do_continue(self, arg)
        finally:
            self.shutdown()

    do_c = do_cont = do_continue

    def do_EOF(self, arg):
        try:
            return Pdb.do_EOF(self, arg)
        finally:
            self.shutdown()

    def set_trace(self, frame):
        assert frame is not None
        try:
            Pdb.set_trace(self, frame)
        except IOError as e:
            if e.errno != errno.ECONNRESET:
                raise e


def set_trace(host=None, port=None):
    host = host or os.environ.get('REMOTE_TCP_HOST', '0.0.0.0')
    port = port or int(os.environ.get('REMOTE_TCP_PORT', '0'))

    if not isinstance(port, int):
        raise ValueError("Port should be an integer!")

    rPdb = RemotePdb(host=host, port=port)
    rPdb.set_trace(frame=sys._getframe().f_back)