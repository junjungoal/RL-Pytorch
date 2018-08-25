
# encoding: utf-8
from __future__ import print_function, division
import numpy as np
import math
from operator import __index__
from multiprocessing import Pipe, Process, Array
import ctypes
import tempfile
import zlib
import struct
import threading
import queue
from PIL import Image


class DecompThread(threading.Thread):
    QUEUE = queue.Queue()

    def run(self):
        while True:
            q = self.QUEUE.get()
            q()

    @classmethod
    def submit(cls, rec):
        ev = threading.Event()

        def run():
            ev.ret = zlib.decompress(rec)
            ev.set()

        cls.QUEUE.put(run)
        return ev


class PrioritizedReplayBuffer:
    THREAD_STARTED = False
    # todo: adjust buf size
    BUFSIZE = -1  # 100*1024*1024

    def __init__(self, action_space_size, state_size,
                 imsize, random_step, total_step,
                 buffer_size=1e5, partition_num=32, init_beta=0.5):
        self._f = tempfile.TemporaryFile()
        self._toc = {}

        self._size = int(buffer_size)
        self._partition_num = partition_num

        self._random_step = random_step
        self._total_step = total_step

        self._action_space_shape = list(action_space_size)
        self._state_space_shape = list(state_size)

        self._imsize = imsize

        self._action_space_size = np.prod(action_space_size) if hasattr(
            action_space_size, "__getitem__") else action_space_size

        self._state_size = np.prod(state_size) if hasattr(state_size, "__getitem__") else state_size

        self._size_prestate = self._state_size * np.float32().itemsize
        self._size_action = self._action_space_size * np.float32().itemsize
        self._size_reward = np.float32().itemsize
        self._size_state = self._state_size * np.float32().itemsize
        self._size_terminal = self._state_size * 1

        self._pos_action = 0 + self._size_prestate
        self._pos_reward = self._pos_action + self._size_action
        self._pos_state = self._pos_reward + self._size_reward
        self._pos_terminal = self._pos_state + self._size_state

        self._priority_queue = {}
        self._distributions = None

        self._init_beta = init_beta
        self._beta_grad = (1. - init_beta) / (total_step - random_step)

        if not PrioritizedReplayBuffer.THREAD_STARTED:
            # todo: adjust number of threads
            for i in range(4):
                d = DecompThread()
                d.daemon = True
                d.start()
            PrioritizedReplayBuffer.THREAD_STARTED = True
        self._index = 0
        self._full = False

    def store(self, prestate, action, reward, state, terminal):
        self.add(prestate, action, reward, state, terminal)
        if self._size < self._index:
            self._full = True
        if not self._full:
            self._index += 1

    def upheap(self, i):
        if i > 1:
            parent_id = math.floor(i/2)
            if self._priority_queue[parent_id] < self._priority_queue[i]:
                tmp = self._priority_queue[i]
                self._priority_queue[i] = self._priority_queue[parent_id]
                self._priority_queue[parent_id] = tmp
                tmp_toc = self._toc[i]
                self._toc[i] = self.toc[parent_id]
                self._toc[parent_id] = tmp_toc
                self.upheap(parent_id)

    def downheap(self, i):
        if i > len(self):
            greatest = i
            left, right = i * 2, i * 2 + 1
            if left < len(self) and self._priority_queue[left] > self._priority_queue[right]:
                greatest = left
            if right < len(self) and self._priority_queue[right] > self._priority_queue[left]:
                greatest = right

            if greatest != i:
                tmp = self._priority_queue[i]
                self._priority_queue[i] = self._priority_queue[greatest]
                self._priority_queue[greatest] = tmp
                tmp_toc = self._toc[i]
                self._toc[i] = self.toc[greatest]
                self._toc[greatest] = tmp_toc
                self.downheap(greatest)

    def update(self, priority, index):
        self._priority_queue[index] = priority
        self.upheap(index)
        self.downheap(index)

    def update_priority(self, indices, delta):
        for i in range(indices):


    def add(self, prestate, action, reward, state, terminal):
        priority = self._priority_queue[0] if len(self) > 0 else 1
        self._priority_queue[self._index] = priority
        # todo: reuse buf when overwriting to the same index
        self._f.seek(0, 2)
        start = self._f.tell()
        c = zlib.compressobj(1)
        self._f.write(c.compress(prestate.astype('float32').tobytes()))
        self._f.write(c.compress(action.astype('float32').tobytes()))
        self._f.write(c.compress(struct.pack('f', reward)))
        self._f.write(c.compress(state.astype('float32').tobytes()))
        self._f.write(c.compress(terminal.tobytes()))  # bool
        self._f.write(c.flush())
        end = self._f.tell()
        self._toc[index] = (start, end)

        self.upheap(self._index)

    def _readrec(self, index):
        f, t = self._toc[index]
        self._f.seek(f, 0)
        rec = self._f.read(t - f)
        return rec

    def _unpack(self, buf):
        prestate = np.frombuffer(buf, np.float32, self._state_size, 0)
        action = np.frombuffer(buf, np.float32, self._action_space_size, self._pos_action)
        reward = struct.unpack('f', buf[self._pos_reward:self._pos_reward + self._size_reward])[0]
        state = np.frombuffer(buf, np.float32, self._state_size, self._pos_state)
        terminal = buf[self._pos_terminal]
        return prestate, action, reward, state, terminal

    def get(self, index):
        buf = self._readrec(index)
        buf = zlib.decompress(buf)
        return self._unpack(buf)

    def build_distribution(self, batch_size):
        res = {}
        partition_size = math.floor(self._size / self._partition_num)
        partition_index = 1
        for i in range(partition_size, self._size + 1, partition_size):
            distribution = {}
            # p_i = 1 / rank(i)
            pdf = list(map(lambda x: math.pow(x, -self.alpha), range(1, i+1)))
            pdf_sum = math.fsum(pdf)
            distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
            cdf = np.cumsum(distribution['pdf'])
            strata_ends = {1: 0, self.batch_size + 1: n}
            step = 1. / batch_size
            index = 1
            for s in range(2, batch_size + 1):
                while cdf[index] < step:
                    index += 1
                strata_ends[s] = index
                step += 1. / batch_size
            distribution['strata_ends'] = strata_ends
            res[partition_index] = distribution
            partition_index += 1
        return res

    def get_minibatch(self, batch_size=32, shuffle=True):
        if self._distributions is None:
            self._distributions = self.build_distribution(batch_size)

        dist_index = math.floor(len(self) / self._size * self._partition_num)
        distribution = self._distributions[dist_index]
        partition_size = math.floor(self._size / self._partition_num)
        partition_max = dist_index * partition_size

        rank_list = []
        for i in range(1, batch_size+1):
            index = random.randint(distribution['strata_ends'][i] + 1,
                                   distribution['strata_ends'][i+1])
            rank_list.append(index)

        beta = min(self._init_beta + (self._total_step - self._random_step - 1) * self._beta_grad, 1)
        p = [distribution['pdf'][v - 1] for v in rank_list]
        weights = np.power(self.size * p, -beta)
        max_weight = max(weights)
        weights /= max_weight

        state_shape = [n, ] + self._state_space_shape
        prestates = []
        actions = np.empty((n, self._action_space_size), dtype=np.float32)
        rewards = np.empty(n, dtype=np.float32)
        states = []
        terminals = np.empty(n, dtype=np.bool)

        events = []
        for index in rank_list:
            buf = self._readrec(index)
            events.append(DecompThread.submit(buf))

        for i, ev in enumerate(events):
            ev.wait()
            prestate, action, reward, state, terminal = self._unpack(ev.ret)
            prestates.append(np.array(Image.fromarray(prestate.reshape(self._state_space_shape).astype(np.uint8)).convert('L').resize(self._imsize)))
            actions[i] = action
            rewards[i] = reward
            states.append(np.array(Image.fromarray(state.reshape(self._state_space_shape).astype(np.uint8)).convert('L').resize(self._imsize)))
            terminals[i] = terminal

        return np.array(prestates, dtype=np.float32), actions, rewards, np.array(states, dtype=np.float32), terminals, rank_list, weights

    def __len__(self):
        return self._index if not self._full else self._size


memmap_path = ".replay_buf"
