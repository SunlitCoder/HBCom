#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading

METEOR_JAR = 'meteor-1.5.jar'


class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        # self.meteor_cmd = ' '.join(self.meteor_cmd)
        # print(os.path.dirname(os.path.abspath(__file__)))
        self.meteor_p = subprocess.Popen(' '.join(self.meteor_cmd),
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         shell=True, encoding='utf-8')
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, preds, refs):
        assert len(preds) == len(refs)
        scores = []
        eval_line = 'EVAL'
        self.lock.acquire()
        for pred, ref in zip(preds, refs):
            # print('Process %d-th item.' % i)
            # print(1)
            assert isinstance(pred, list)
            # assert isinstance(pred[0], int) or isinstance(pred[0], str)
            # assert isinstance(ref, list)
            assert isinstance(ref[0], list)
            pred = ' '.join(pred)
            ref = [' '.join(item) for item in ref]
            stat = self._stat(pred, ref)
            eval_line += ' ||| {}'.format(stat)

        # assert (gts.keys() == res.keys())
        # imgIds = gts.keys()
        # scores = []
        #
        # eval_line = 'EVAL'
        # self.lock.acquire()
        # for i in imgIds:
        #     assert (len(res[i]) == 1)
        #     stat = self._stat(res[i][0], gts[i])
        #     eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        self.meteor_p.stdin.flush()
        for i in range(0, len(preds)):
            # print('process %d-th item.'%i)
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        print(scores)

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        self.meteor_p.stdin.flush()
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        self.meteor_p.stdin.flush()
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
