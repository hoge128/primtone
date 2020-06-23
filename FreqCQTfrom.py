"""
bin表記からspec表記に
"""
# v2020.1.4

# from scipy import arange, complex128, exp, log2, zeros
from numpy import arange, complex128, exp, log2, zeros, ceil, dot, float64
from numpy import pi as mpi
import numpy as np

from scipy.sparse import lil_matrix, csr_matrix

from matplotlib import pylab as pl
import japanize_matplotlib

import math

import librosa
from librosa.display import specshow
from librosa import load,pseudo_cqt
from librosa.effects import harmonic
from librosa.util import find_files
import matplotlib.pyplot as plt

from tqdm import tqdm
import GenTrueCQT as gtc

# 2020.1.4
# np3CQTとkrCQTとMEL_Hの比較
# FILENAME = "/home/ito/work/_ALLDATA/KIZAKI_218/Datas/kro3CQT/01_-_Please_Please_Me/01_-_I_Saw_HerStandingThere.wav.npy"
# FILENAME = "/home/ito/work/_ALLDATA/NAKAYAMA_318/Datas/np3CQT/fold2/01_-_Please_Please_Me/01_-_I_Saw_HerStandingThere.wav.npy"
# FILENAME = "/home/ito/work/_ALLDATA/NAKAYAMA_318/Datas/MEL_H/fold0/01_-_Please_Please_Me/11_-_Do_You_Want_To_Know_A_Secret.wav.npy"

FILENAME = "/home/ito/work/_ALLDATA/NAKAYAMA_318/Datas/MEL_H/fold2/01_-_Please_Please_Me/01_-_I_Saw_HerStandingThere.wav.npy"
# FILENAME = "/home/ito/work/_ALLDATA/NAKAYAMA_318/Datas/CQT_H/fold2/01_-_Please_Please_Me/01_-_I_Saw_HerStandingThere.wav.npy"
"""
複数チャンネルある場合の表記
"""
# LAYER = 2
# cqt = cqt[LAYER]
# cqt = np.transpose(cqt, (1, 0))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
CQT1チャンネル (1, frame, bin)  ==> (bin, frame)
"""
# cqt = np.reshape(cqt, (cqt.shape[1], cqt.shape[2]))
# cqt = np.transpose(cqt, (1, 0))
""""""""""""""""""""""""""""""""""""""""""""""""

"""
RE or MY
"""
# cqt = cqt[0]
# cqt = np.transpose(cqt, (1, 0))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def main():
    """
    chord = np.load("Cov4.npy")
    # chord = np.zeros(4 * 1389 * 144).reshape(4, 1389, 144)   # 空データを入れた場合
    fn1 = np.load("C_ov4.npy")
    fn2 = np.load("E_ov4.npy")
    # fn2 = np.zeros(4 * 1389 * 144).reshape(4, 1389, 144)
    fn3 = np.load("G_ov4.npy")
    # chordBoard(chord, fn1, fn2, fn3, analysisPoint_input=12, fn="Cメジャーの倍音構造", FLAG_ylimit=False)
    # cqtV2_mix_chord(chord, fn1, fn2, fn3, analysisPoint_input=5, fn="Cメジャーの倍音構造", FLAG_ylimit=False, FLAG_log=True)
    """

    cqtV2(analysisPoint_input=1111, fn="MEL_H_log"+"01_-_I_Saw_HerStandingThere", FLAG_ylimit=False, FLAG_log=True)
    

def view1():
    cqt = np.load(FILENAME)
    fline = gtc.GenMusicalScale()
    endpoint = cqt.shape[1]
    analysisPoint = 1203
    f = cqt[:,analysisPoint]
    f = math.log10(f)
    # plt.xlim([0,1000.0])
    plt.ylim([0,0.25])
    # plt.plot(fline, f)
    plt.title("第{}層 monowav-第{}/{}フレーム\nCQT中心0~1000Hzまでの様子".format(LAYER+1, analysisPoint, endpoint))
    plt.xlabel('Frequency(fCQT)', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid()
    plt.show()


def cqtV2(cqt = None, analysisPoint_input = None, fn=None, FLAG_ylimit = True, FLAG_log = False):
    cqt_v2 = np.load(FILENAME) if cqt is None else cqt
    titlesrc = FILENAME[1:40] if fn is None else fn
    print(cqt_v2.shape)
    '''解析フレーム選択'''
    analysisPoint = 2222 # きれいなところ 多分 01_-_I_Saw_HerStandingThere.wav
    # analysisPoint = 2164

    if analysisPoint is not None:
        analysisPoint = analysisPoint_input

    # YMAX = 10     # log10用
    YMAX = 0.03     # 普通の
    
    
    plt.figure()
    # fline = gtc.GenMusicalScale()
    fline = np.arange(384)

    '''1層目'''
    plt.subplot(311)
    lay = 0
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(0.0001 + f) if FLAG_log else f
    endpoint = cqt_.shape[1]
    plt.title("{} :第{}フレーム".format(titlesrc, analysisPoint))
    plt.plot(fline, f, label="第1層(CQT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)

    '''2層目'''
    plt.subplot(312)
    lay+=1
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(0.0001 + f) if FLAG_log else f
    plt.plot(fline, f, label="第2層(FFT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)
    plt.ylabel('Amplitude', fontsize=20)


    '''3層目'''
    plt.subplot(313)
    lay+=1
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(0.0001 + f) if FLAG_log else f
    plt.plot(fline, f, label="第3層(FFT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    plt.xlabel('Frequency(fCQT)', fontsize=20)
    leg = plt.legend(loc=1, fontsize=15)


    plt.show()


def cqtV2_for2ch():
    cqt_v2 = np.load(FILENAME)
    print(cqt_v2.shape)
    '''解析フレーム選択'''
    analysisPoint = 2222 # きれいなところ 多分 01_-_I_Saw_HerStandingThere.wav
    # analysisPoint = 2164


    FLAG_log = False
    FLAG_ylimit = False
    YMAX = 0.20
    YMAX = 10
    
    plt.figure()
    fline = gtc.GenMusicalScale()

    '''1層目'''
    plt.subplot(211)
    lay = 0
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(f) if FLAG_log else f
    endpoint = cqt_.shape[1]
    plt.title("{} :第{}フレーム".format(FILENAME[1:40], analysisPoint))
    plt.plot(fline, f, label="第1層(CQT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)

    '''2層目'''
    plt.subplot(212)
    lay+=1
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(f) if FLAG_log else f
    plt.plot(fline, f, label="第2層(FFT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)
    plt.ylabel('Amplitude', fontsize=20)
    plt.xlabel('Frequency(fCQT)', fontsize=20)

    plt.show()


def cqtV2_for4ch(cqt = None, analysisPoint_input = None, fn=None, FLAG_ylimit = True, FLAG_log = False):
    cqt_v2 = np.load(FILENAME) if cqt is None else cqt
    print(cqt_v2.shape)
    '''解析フレーム選択'''
    analysisPoint = 2222 # きれいなところ 多分 01_-_I_Saw_HerStandingThere.wav
    # analysisPoint = 2164

    if analysisPoint is not None:
        analysisPoint = analysisPoint_input

    # YMAX = 10     # log10用
    YMAX = 0.05     # 普通の
    
    
    plt.figure()
    fline = gtc.GenMusicalScale()

    '''1層目'''
    plt.subplot(411)
    lay = 0
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(f) if FLAG_log else f
    endpoint = cqt_.shape[1]
    FN = FILENAME[1:40] if fn is None else fn
    plt.title("{} :第{}フレーム".format(FN, analysisPoint))
    plt.plot(fline, f, label="第1層(CQT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)

    '''2層目'''
    plt.subplot(412)
    lay+=1
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(f) if FLAG_log else f
    plt.plot(fline, f, label="第2層(FFT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)
    plt.ylabel('Amplitude', fontsize=20)


    '''3層目'''
    plt.subplot(413)
    lay+=1
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(f) if FLAG_log else f
    plt.plot(fline, f, label="第3層(FFT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)


    '''4層目'''
    plt.subplot(414)
    lay+=1
    cqt_ = cqt_v2[lay]
    cqt_ = np.transpose(cqt_, (1, 0))
    f = cqt_[:,analysisPoint]
    f = np.log10(f) if FLAG_log else f
    plt.plot(fline, f, label="第4層(FFT)")
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    plt.xlabel('Frequency(fCQT)', fontsize=20)
    leg = plt.legend(loc=1, fontsize=15)


    plt.show()


def chordBoard(chord, fn1, fn2, fn3, analysisPoint_input = None, fn=None, FLAG_ylimit = True, FLAG_log = False):
    '''解析フレーム選択'''
    analysisPoint = 2222 # きれいなところ 多分 01_-_I_Saw_HerStandingThere.wav
    # analysisPoint = 2164

    if analysisPoint is not None:
        analysisPoint = analysisPoint_input

    # YMAX = 10     # log10用
    YMAX = 0.05     # 普通の
    
    c0, c1,c2,c3 = "red", "green","black","blue"      # 各プロットの色
    l0, l1,l2,l3 = "Cメジャー", "C", "E", "G"
    FN = "Cメジャーが構成する倍音構造"
    plt.figure()
    fline = gtc.GenMusicalScale()

    lay = 0
    sploting = 411
    for i in range(4):
        plt.subplot(sploting)
        chord_, fn1_, fn2_, fn3_= chord[lay], fn1[lay], fn2[lay], fn3[lay]
        chord_ = np.transpose(chord_, (1, 0))
        fn1_ = np.transpose(fn1_, (1, 0))
        fn2_ = np.transpose(fn2_, (1, 0))
        fn3_ = np.transpose(fn3_, (1, 0))
        chord_ = chord_[:,analysisPoint]
        fn1_ = fn1_[:,analysisPoint]
        fn2_ = fn2_[:,analysisPoint]
        fn3_ = fn3_[:,analysisPoint]
        chord_ = np.log10(chord_) if FLAG_log else chord_
        fn1_ = np.log10(fn1_) if FLAG_log else fn1_
        fn2_ = np.log10(fn2_) if FLAG_log else fn2_
        fn3_ = np.log10(fn3_) if FLAG_log else fn3_
        endpoint = chord.shape[1]
        if sploting == 414:
            plt.title("{} :第{}フレーム".format(FN, analysisPoint))
        plt.plot(fline, chord_, color = c0, label=l0)
        plt.plot(fline, fn1_, color = c1, label=l1, linestyle = "-", alpha=0.6)
        plt.plot(fline, fn2_, color = c2, label=l2, linestyle = "--", alpha=0.6)
        plt.plot(fline, fn3_, color = c3, label=l3, linestyle = ":", alpha=0.6)

        plt.ylim([0, YMAX]) if FLAG_ylimit else None
        plt.grid()
        leg = plt.legend(loc=1, fontsize=10)

        sploting += 1
    plt.show()

"""
np.load(でロード済みのものを渡す。)
chord, fn1, fn2, fn3, analysisPoint_input = None, fn=None, FLAG_ylimit = True, FLAG_log = False):
"""
def cqtV2_mix_chord(chord, fn1, fn2, fn3, analysisPoint_input = None, fn=None, FLAG_ylimit = True, FLAG_log = False):
    '''解析フレーム選択'''
    analysisPoint = 2222 # きれいなところ 多分 01_-_I_Saw_HerStandingThere.wav
    # analysisPoint = 2164

    if analysisPoint is not None:
        analysisPoint = analysisPoint_input

    # YMAX = 10     # log10用
    YMAX = 0.05     # 普通の
    
    c0, c1,c2,c3 = "red", "green","black","blue"      # 各プロットの色
    l0, l1,l2,l3 = "Cメジャー", "C", "E", "G"
    FN = "Cメジャーが構成する倍音構造"
    plt.figure()
    fline = gtc.GenMusicalScale()




    '''1層目'''
    plt.subplot(411)
    lay = 0
    chord_, fn1_, fn2_, fn3_= chord[lay], fn1[lay], fn2[lay], fn3[lay]
    chord_ = np.transpose(chord_, (1, 0))
    fn1_ = np.transpose(fn1_, (1, 0))
    fn2_ = np.transpose(fn2_, (1, 0))
    fn3_ = np.transpose(fn3_, (1, 0))

    chord_ = chord_[:,analysisPoint]
    fn1_ = fn1_[:,analysisPoint]
    fn2_ = fn2_[:,analysisPoint]
    fn3_ = fn3_[:,analysisPoint]
    chord_ = np.log10(chord_) if FLAG_log else chord_
    fn1_ = np.log10(fn1_) if FLAG_log else fn1_
    fn2_ = np.log10(fn2_) if FLAG_log else fn2_
    fn3_ = np.log10(fn3_) if FLAG_log else fn3_
    endpoint = chord.shape[1]
    plt.title("{} :第{}フレーム".format(FN, analysisPoint))
    # plt.plot(fline, chord_, color = c0, label=l0, linewidth = 3.0, alpha=0.6)
    plt.plot(fline, fn1_, color = c1, label=l1, linestyle = "-", alpha=0.6)
    # plt.plot(fline, fn2_, color = c2, label=l2, linestyle = "--", alpha=0.6)
    plt.plot(fline, fn3_, color = c3, label=l3, linestyle = ":", alpha=0.6)
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)

    '''2層目'''
    plt.subplot(412)
    lay+=1
    chord_, fn1_, fn2_, fn3_= chord[lay], fn1[lay], fn2[lay], fn3[lay]
    chord_ = np.transpose(chord_, (1, 0))
    fn1_ = np.transpose(fn1_, (1, 0))
    fn2_ = np.transpose(fn2_, (1, 0))
    fn3_ = np.transpose(fn3_, (1, 0))

    chord_ = chord_[:,analysisPoint]
    fn1_ = fn1_[:,analysisPoint]
    fn2_ = fn2_[:,analysisPoint]
    fn3_ = fn3_[:,analysisPoint]
    chord_ = np.log10(chord_) if FLAG_log else chord_
    fn1_ = np.log10(fn1_) if FLAG_log else fn1_
    fn2_ = np.log10(fn2_) if FLAG_log else fn2_
    fn3_ = np.log10(fn3_) if FLAG_log else fn3_

    # plt.plot(fline, chord_, color = c0, label=l0, linewidth = 3.0, alpha=0.6)
    plt.plot(fline, fn1_, color = c1, label=l1, linestyle = "-", alpha=0.6)
    # plt.plot(fline, fn2_, color = c2, label=l2, linestyle = "--", alpha=0.6)
    plt.plot(fline, fn3_, color = c3, label=l3, linestyle = ":", alpha=0.6)
    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)
    plt.ylabel('Amplitude', fontsize=20)


    '''3層目'''
    plt.subplot(413)
    lay+=1
    
    chord_, fn1_, fn2_, fn3_= chord[lay], fn1[lay], fn2[lay], fn3[lay]
    chord_ = np.transpose(chord_, (1, 0))
    fn1_ = np.transpose(fn1_, (1, 0))
    fn2_ = np.transpose(fn2_, (1, 0))
    fn3_ = np.transpose(fn3_, (1, 0))

    chord_ = chord_[:,analysisPoint]
    fn1_ = fn1_[:,analysisPoint]
    fn2_ = fn2_[:,analysisPoint]
    fn3_ = fn3_[:,analysisPoint]
    chord_ = np.log10(chord_) if FLAG_log else chord_
    fn1_ = np.log10(fn1_) if FLAG_log else fn1_
    fn2_ = np.log10(fn2_) if FLAG_log else fn2_
    fn3_ = np.log10(fn3_) if FLAG_log else fn3_

    # plt.plot(fline, chord_, color = c0, label=l0, linewidth = 3.0, alpha=0.6)
    plt.plot(fline, fn1_, color = c1, label=l1, linestyle = "-", alpha=0.6)
    # plt.plot(fline, fn2_, color = c2, label=l2, linestyle = "--", alpha=0.6)
    plt.plot(fline, fn3_, color = c3, label=l3, linestyle = ":", alpha=0.6)

    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    leg = plt.legend(loc=1, fontsize=15)


    '''4層目'''
    plt.subplot(414)
    lay+=1

    chord_, fn1_, fn2_, fn3_= chord[lay], fn1[lay], fn2[lay], fn3[lay]
    chord_ = np.transpose(chord_, (1, 0))
    fn1_ = np.transpose(fn1_, (1, 0))
    fn2_ = np.transpose(fn2_, (1, 0))
    fn3_ = np.transpose(fn3_, (1, 0))

    chord_ = chord_[:,analysisPoint]
    fn1_ = fn1_[:,analysisPoint]
    fn2_ = fn2_[:,analysisPoint]
    fn3_ = fn3_[:,analysisPoint]
    chord_ = np.log10(chord_) if FLAG_log else chord_
    fn1_ = np.log10(fn1_) if FLAG_log else fn1_
    fn2_ = np.log10(fn2_) if FLAG_log else fn2_
    fn3_ = np.log10(fn3_) if FLAG_log else fn3_

    # plt.plot(fline, chord_, color = c0, label=l0, linewidth = 3.0, alpha=0.6)
    plt.plot(fline, fn1_, color = c1, label=l1, linestyle = "-", alpha=0.6)
    # plt.plot(fline, fn2_, color = c2, label=l2, l44100inestyle = "--", alpha=0.6)
    plt.plot(fline, fn3_, color = c3, label=l3, linestyle = ":", alpha=0.6)

    plt.ylim([0, YMAX]) if FLAG_ylimit else None
    plt.grid()
    plt.xlabel('Frequency(fCQT)', fontsize=20)
    leg = plt.legend(loc=1, fontsize=15)


    plt.show()


if __name__ == "__main__":
    main()
    
