#coding=utf-8
# V:3.0.0
# 最終更新
# exexOvertone完成です。
# 2.2.3: overfreqが"hann_all_加算モードになりました。これが一番CQTとFFTのレンジの差が狭まります。
# 3.0.0: powerVer。無理やり合わせるため各所に倍数制御あり。
"""
Constant-Q Transform
Calculation of a constant Q spectral transform を元に
python3.5ベースで実装したもの
計算はかなり遅いので、GPUを使用してむりやり高速化したものもある。

http://yukara-13.hatenablog.com/entry/2013/12/01/222742
"""
# 複素数計算系
from numpy import arange, complex128, exp, log2, zeros
from numpy import pi as mpiplottool
import numpy as np

from numpy import pi as mpi

from scipy.sparse import lil_matrix, csr_matrix

import math
import os
import sys
import librosa
from librosa import load,pseudo_cqt
from librosa.effects import harmonic
from librosa.util import find_files
from tqdm import tqdm
import datetime

# # 自分の提案する最も良いパラメータ
# SR = 44100
# SAMPLE_MUM = 8192
# HOP = 128
# NUM_OCT = 6


'''
グローバル変数群
これらを変更することで、
'''

SR = 22050
SAMPLE_MUM = 4096
CQTHOP = 1024
PATH_TRUECQT= "Datas/krCQT/"
PATH_AUDIO = "Datas/Audio/"
PATH_OVERTONEmix2 = "Datas/kro2CQT"
PATH_OVERTONEmix3 = "Datas/kro3CQT"
PATH_OVERTONEmix4 = "Datas/kro4CQT"


PATH_ov2_allcre = "Datas/ovc2CQT"
PATH_ov3_allcre = "Datas/ovc3CQT"
PATH_ov4_allcre = "Datas/ovc4CQT"

PATH_hrCQT = "Datas/hrCQT/" # harmonicを追加したもの2019.12.21
PATH_hrov2CQT = "Datas/hr2CQT" # そのCQTにさらに倍音もharmonic
PATH_hrov3CQT = "Datas/hr3CQT" # そのCQTにさらに倍音もharmonic

PATH_pwCQT = "Datas/pwCQT/"     # powerスペクトル
PATH_pwov2CQT = "Datas/pw2CQT"
PATH_pwov3CQT = "Datas/pw3CQT"

PATH_SAME2 = "Datas/same2CQT"
PATH_SAME3 = "Datas/same3CQT"


two_pi_j = 2 * mpi * 1j     # 後で治すけど、これの定義イランクネ？

# 1オクターブをどのくらいで解析をするのか
note_resolution = 24

# q_rate_def = 20. * note_resolution # qrate
q_rate_def = 1.

'''補正係数'''
ACF_hann = 0.5
PCF_hann = 0.375
ENBCF_hann = 1.5

ACF_hamm = 0.54
PCF_hamm = 0.3974
ENBCF_hamm = 1.362825


def main():
    # makeCQTexOvertone()
    # debug_makeSoloData()
    # debug()
    # makeCQTData()
    baka()

def shuffleCQT_baka():
    TORIDASUBASHO = PATH_hrCQT
    HOZONBASYO = PATH_SAME2
    HOZONBASYO2 = PATH_SAME3

    OVTONE = 2

    audiolist = find_files(PATH_AUDIO, ext="wav")
    cqtlist = find_files(TORIDASUBASHO, ext="npy")
    itemlist = len(audiolist)
    i_counter = 1
    for audiofile, cqtfile in zip(audiolist, cqtlist):
        print("{}/{}".format(i_counter, itemlist))
        filename = audiofile.split('/')[-1]
        albname = audiofile.split('/')[-2]
        # foldname = audiofile.split('/')[-3]
        cqt_filename = cqtfile.split("/")[-1]
        if (filename.split(".")[0] != cqt_filename.split(".")[0]):
            print("file_not_match", filename, cqt_filename)
# ディレクトリチェック

        if not (os.path.exists(HOZONBASYO+'/'+albname)):
            os.makedirs(HOZONBASYO+'/'+albname)
        if not (os.path.exists(HOZONBASYO+'/'+albname + '/'+filename+'.npy')):
            cqt = np.load(cqtfile)
            excqt = np.vstack((cqt, cqt))
            np.save(HOZONBASYO+'/'+albname+'/'+filename+'.npy', np.array(excqt, dtype="float32"))

        if not (os.path.exists(HOZONBASYO2+'/'+albname)):
            os.makedirs(HOZONBASYO2+'/'+albname)
        if not (os.path.exists(HOZONBASYO2+'/'+albname + '/'+filename+'.npy')):
            cqt = np.load(cqtfile)
            excqt = np.vstack((cqt, cqt, cqt))
            np.save(HOZONBASYO2+'/'+albname+'/'+filename+'.npy', np.array(excqt, dtype="float32"))

        i_counter += 1



def baka():
    TORIDASUBASHO = PATH_hrCQT
    HOZONBASYO = PATH_SAME2
    HOZONBASYO2 = PATH_SAME3

    OVTONE = 2

    audiolist = find_files(PATH_AUDIO, ext="wav")
    cqtlist = find_files(TORIDASUBASHO, ext="npy")
    itemlist = len(audiolist)
    i_counter = 1
    for audiofile, cqtfile in zip(audiolist, cqtlist):
        print("{}/{}".format(i_counter, itemlist))
        filename = audiofile.split('/')[-1]
        albname = audiofile.split('/')[-2]
        # foldname = audiofile.split('/')[-3]
        cqt_filename = cqtfile.split("/")[-1]
        if (filename.split(".")[0] != cqt_filename.split(".")[0]):
            print("file_not_match", filename, cqt_filename)
# ディレクトリチェック

        if not (os.path.exists(HOZONBASYO+'/'+albname)):
            os.makedirs(HOZONBASYO+'/'+albname)
        if not (os.path.exists(HOZONBASYO+'/'+albname + '/'+filename+'.npy')):
            cqt = np.load(cqtfile)
            excqt = np.vstack((cqt, cqt))
            np.save(HOZONBASYO+'/'+albname+'/'+filename+'.npy', np.array(excqt, dtype="float32"))

        if not (os.path.exists(HOZONBASYO2+'/'+albname)):
            os.makedirs(HOZONBASYO2+'/'+albname)
        if not (os.path.exists(HOZONBASYO2+'/'+albname + '/'+filename+'.npy')):
            cqt = np.load(cqtfile)
            excqt = np.vstack((cqt, cqt, cqt))
            np.save(HOZONBASYO2+'/'+albname+'/'+filename+'.npy', np.array(excqt, dtype="float32"))

        i_counter += 1

def debug_makeSoloData():
    audiofile = "01_-_I_Saw_HerStandingThere.wav"
    # audiofile = "1046.502Hz2093.004Hz3139.506Hz4186.008Hz.wav"
    wav, sr = load(audiofile, sr=SR)
    cqt_spec, freqs = cqt(wav, sr, fmin="C1")
    # cqt_power = np.abs(cqt_spec)                # absとそうじゃないやつの違い
    cqt_power = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in cqt_spec], dtype="float32")
    cqt_power = np.power(cqt_power, 2)
    cqt_power *= 2
    cqt_power = cqt_power.reshape(1,cqt_power.shape[0],cqt_power.shape[1])
    np.save("01ISAWHERSTAND.npy", np.array(cqt_power, dtype="float32"))


def debug():
    audiofile = "01_-_I_Saw_HerStandingThere.wav"
    # audiofile = "1046.502Hz2093.004Hz3139.506Hz4186.008Hz.wav"

    wav, sr = load(audiofile, sr=SR)

    """倍音生成"""
    mokuteki = "power"
    cqt = np.load("01ISAWHERSTAND.npy")
    tmp = exOvertone(cqt, wav, overtone=2)
    last = exOvertone(tmp, wav, overtone=3)
    np.save("debug_tmp.npy", np.array(last, dtype="float32"))
    # last = np.load("debug_tmp.npy")
    aa = 2222

    import ShowPrintFrame as fff
    fff.View3ch(last,aa)
    import FreqCQTfrom as fcf
    fcf.cqtV2(last, aa, FLAG_ylimit=False)

"""
2倍音まで込み込みのやつ。
"""
def makeCQTexOvertone():
    TORIDASUBASHO = PATH_pwCQT
    HOZONBASYO = PATH_pwov2CQT # n倍音を保存するディレクトリ
    OVTONE = 2

    audiolist = find_files(PATH_AUDIO, ext="wav")
    cqtlist = find_files(TORIDASUBASHO, ext="npy")
    itemlist = len(audiolist)
    i_counter = 1
    for audiofile, cqtfile in zip(audiolist, cqtlist):
        print("{}/{}".format(i_counter, itemlist))
        filename = audiofile.split('/')[-1]
        albname = audiofile.split('/')[-2]
        # foldname = audiofile.split('/')[-3]
        cqt_filename = cqtfile.split("/")[-1]
        if (filename.split(".")[0] != cqt_filename.split(".")[0]):
            print("file_not_match", filename, cqt_filename)
# ディレクトリチェック

        if not (os.path.exists(HOZONBASYO+'/'+albname)):
            os.makedirs(HOZONBASYO+'/'+albname)
        if not (os.path.exists(HOZONBASYO+'/'+albname + '/'+filename+'.npy')):
            wav, sr = load(audiofile, sr=SR)
            cqt = np.load(cqtfile)
            excqt = exOvertone(cqt, wav, overtone=OVTONE)
            np.save(HOZONBASYO+'/'+albname+'/'+filename+'.npy', np.array(excqt, dtype="float32"))
        i_counter += 1





def makeCQTData():
    SAVEDIR = PATH_pwCQT

    audiolist = find_files(PATH_AUDIO, ext="wav")
    itemlist = len(audiolist)
    for i, audiofile in enumerate(audiolist):
        print("{}/{}".format(i+1, itemlist))
        wav, sr = load(audiofile, sr=SR)
        filename = audiofile.split('/')[-1]
        albname = audiofile.split('/')[-2]
        # foldname = audiofile.split('/')[-3]
        # ディレクトリチェック
        if not (os.path.exists(SAVEDIR+'/'+albname)):
            os.makedirs(SAVEDIR+'/'+albname)
        if not (os.path.exists(SAVEDIR+'/'+albname + '/'+filename+'.npy')):
            cqt_spec, freqs = cqt(wav, sr, fmin="C1")

            # cqt_power = np.abs(cqt_spec)  # sqrtモードで行こう # 2019.12.22
            cqt_power = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in cqt_spec], dtype="float32")
            cqt_power = np.power(cqt_power, 2) # power 系1
            cqt_power *= 2                     # power 系2

            cqt_power = cqt_power.reshape(1,cqt_power.shape[0],cqt_power.shape[1])
            np.save(SAVEDIR+'/'+albname+'/'+filename+'.npy', np.array(cqt_power, dtype="float32"))


#######################################################################################################
# 
# 基本関数群
# 
#######################################################################################################
"""
CQTのbinの生成(平均律音階)
1. fmin_note [str] : 平均律音階の基準音
2. bin_per_oct [int] : 1オクターブを分割する数
3. num_oct [int] : オクターブを繰り返す回数

return [list]
(bin_per_oct) x (num_oct)分の平均律音階が記録された配列 
"""
def GenMusicalScale(fmin_note=None, bin_per_oct=24, num_oct=6):
    if fmin_note is None:
        fmin = librosa.note_to_hz("C1")
    else: 
        fmin = librosa.note_to_hz(str(fmin_note))
    r = 1. / bin_per_oct
    cm = math.pow(2,r)
    return fmin * cm**arange(num_oct * bin_per_oct)


# # 自作ハミング窓
# def hammingWindow(sample):
#     n = np.arange(0, sample)  # numpy arrayはmath関数に渡せないorz
#     _window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n/(sample-1) )
#     return _window
"""
窓関数をかけるための関数
numpyの実装は端点において、実装がずれているのでこちらを使用するのが良い。
⚠⚠⚠ ハミング窓だと思っていたものはどうやら、ハニング窓だった件。実装ミス ⚠⚠⚠

1. sample [int] : 窓の範囲
2. mode [str] : 
    "hann" : hann window ハン窓、ハニング窓で窓掛けを行う。
    "hamm" : hamming window, ハミング窓をで窓掛けを行う。
return 
sampleサイズ分の窓掛けされたnp.arrayが返り値となります。
"""
def hWindow(sample, mode="hann"):
    n = np.arange(0, sample)  # numpy arrayはmath関数に渡せないorz
    if mode is "hann":
        _window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n/(sample-1) )
    elif "hamm":
        _window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n/(sample-1) )
    else:
        print("正確に窓を指定してください。")
        _window = None
    return _window


def hammingWindow(sample):
    return hWindow(sample, mode = "hamm")


def hanningWindow(sample):
    return hWindow(sample, mode = "hann")

#######################################################################################################
# 
# CQTの変換
# 
#######################################################################################################


"""
元論文1991の実装。
かなり遅い。
中身を理解するにはこいつのほうがいい。
"""
def cqt(time_signal, f_s, q_rate = q_rate_def, fmin=None, fratio = note_resolution, num_oct=6):
    # fminの定義
    if fmin is None:
        fmin = librosa.note_to_hz("C1")
                       
    nhop = CQTHOP

    # Calculate Constant-Q Properties
    freqs = GenMusicalScale(fmin, note_resolution, num_oct) # 各周波数ビンの中心周波数
    nfreq = int(num_oct * note_resolution)                  # 周波数ビンの個数定義
    Q = int((1. / ((2 ** (1. / fratio)) - 1)) * q_rate)     # Eq.(2) Q Value from 1992
    

    sig_len = len(time_signal)                              # サンプル数
    nframe = int(sig_len / nhop)                            # フレーム数

    ret = zeros([nframe, nfreq], dtype = complex128)        # Constant-Q spectrogram

    # X[k] = 1/N[k] Σ W[k,n] x[n]   exp{-j2πQn/N[k]} ・・・(5) from 1992
    # ______⎣  ①  ⎦__⎣ ②  ⎦_⎣ ③ ⎦_⎣ ④             ⎦
    for k in tqdm(range(nfreq)):
        freq = freqs[k]
        nsample = int(round(float(f_s * Q) / freq))         # Eq.(3) N_sample from 1992
        hsample = int(nsample / 2)                          # Sample の半分

        # Calculate window function (and weight).
        phase = exp(-two_pi_j * Q * arange(nsample, dtype = float) / nsample) # Eq. (5) from 1992 ④
        # weight = phase * hammingWindow(nsample)                               # Eq. (5) from 1992 ④x②=> ㈹
        weight = phase * hanningWindow(nsample)                               # Eq. (5) from 1992 ④x②=> ㈹   #2019.12.17 hamm => hann


        # Perform Constant-Q Transform.
        for iiter in range(nframe):
            iframe = iiter
            istart = iframe * nhop - hsample   # Where i番目のフレーム  始まる
            iend = istart + nsample            #  Where i番目のフレーム  終わり
            sig_start = min(max(0, istart), sig_len)
            sig_end = min(max(0, iend), sig_len)
            win_start = min(max(0, sig_start - istart), nsample)  # padd
            win_end = min(max(0, sig_len - istart), nsample)      # padd
            win_slice = weight[win_start : win_end]
            y = time_signal[sig_start:sig_end]
            ret[iiter, k] = (y * win_slice).sum() / nsample       # Eq. (5) from 1992 ㈹x③x①

    return ret, freqs


def fcqt(time_signal, fs, q_rate = q_rate_def, fmin = None, fratio = note_resolution, spThresh = 0.0054,  num_oct=6):
     # fminの定義
    if fmin is None:
        fmin = librosa.note_to_hz("C1")
    
    # フレーム移動量は固定
    nhop = CQTHOP

    # Calculate Constant-Q Properties
    freqs = GenMusicalScale(fmin, note_resolution, num_oct) # 各周波数ビンの中心周波数
    nfreq = int(num_oct * note_resolution)                  # 周波数ビンの個数定義
    Q = int((1. / ((2 ** (1. / fratio)) - 1)) * q_rate)     # Eq.(2) Q Value from 1992

    sig_len = len(time_signal)                              # サンプル数
    nframe = int(sig_len / nhop)                            # フレーム数


    # N  > max(N_k)
    fftLen = int(2 ** (ceil(log2(int(float(fs * Q) / freqs[0])))))      # 
    h_fftLen = fftLen / 2

    fftLen = int(2 ** (ceil(log2(int(float(fs * Q) / freqs[0])))))
    h_fftLen = int(fftLen / 2)
   
    # ===================
    #  カーネル行列の計算
    # ===================
    sparseKernel = zeros([nfreq, fftLen], dtype = complex128)
    for k in range(nfreq):
        tmpKernel = zeros(fftLen, dtype = complex128)
        freq = freqs[k]
        # N_k 
        N_k = int(float(fs * Q) / freq)
        # FFT窓の中心を解析部分に合わせる．
        startWin = int((fftLen - N_k) / 2)
        tmpKernel[startWin : startWin + N_k] = (hammingWindow(N_k) / N_k) * exp(two_pi_j * Q * arange(N_k, dtype = float64) / N_k)
        # FFT (kernel matrix)
        sparseKernel[k] = np.fft.fft(tmpKernel)

    ### 十分小さい値を０にする
    sparseKernel[abs(sparseKernel) <= spThresh] = 0
    
    ### スパース行列に変換する
    sparseKernel = csr_matrix(sparseKernel)
    ### 複素共役にする
    sparseKernel = sparseKernel.conjugate() / fftLen
 

    # ===========
    #  Execution
    # ===========
    ### New signal (for Calculation)
    new_sig = zeros(len(time_signal) + fftLen, dtype = float64)
    new_sig[h_fftLen : -h_fftLen] = time_signal
    
    ret = zeros([nframe, nfreq], dtype = complex128)
    for iiter in tqdm(range(nframe)):
        istart = iiter * nhop
        iend = istart + fftLen
        # FFT (input signal)?
        sig_fft = np.fft.fft(new_sig[istart : iend])
        # 行列積?
        ret[iiter] = sig_fft * sparseKernel.T
 
    return ret, freqs


"""
計算した倍音配列をCQT配列の後ろのstack
cqt:[numpy array]
    想定しているcqtは(1, フレーム数, ビン数)
また入力されるCQTのサンプリングレートは十分に考慮されたし、
# SR = 44100
# SAMPLE_MUM = 8192
# 時間分解能 N/f_s = 0.18
# 周波数分解能 f_s / N = 5.38
なので、SR = 22050 のとき、SAMPLE_NUM = 4096のとき、最高性能
"""
def exOvertone(cqt, wav, hop = CQTHOP, overtone=2, sr=SR, sample=SAMPLE_MUM, wingrange = 4, fmin=None, note_resolution = note_resolution, num_oct = 6):
    frame_num = cqt.shape[1]                                   # フレーム回数
    bin_num = cqt.shape[2]                                     # ループ回数
    freqs = GenMusicalScale(fmin, note_resolution, num_oct)    # CQT中心周波数
    fft_freqs = np.fft.fftfreq(sample, 1/sr)
    zeropad = np.zeros(int(sample/2), dtype="float32") 
    hsample = int(sample / 2)                                  # 切り出す長さの指定
    wav_pad = np.hstack((zeropad, wav, zeropad))
    center = hsample
    bind_frame = None       # 1フレームのbind_binをフレーム数分束ねたもの。これを元のCQT配列にvstackする 2019.12.11
    for f_itr in tqdm(range(frame_num)):
        cut = wav_pad[center - hsample : center + hsample]
        wcut = cut * hanningWindow(len(cut))                   # hamming とhanningを変えた
        wcut = wcut / ACF_hann / PCF_hann / ENBCF_hann
        # amp = np.abs(np.fft.fft(a=wcut) / (SAMPLE_MUM/ACF_hann)) # こっち?
        # amp = np.abs(np.fft.fft(a=wcut) / (SAMPLE_MUM)) # これは修正しろ!
        fft_tmp = np.fft.fft(a=wcut) / SAMPLE_MUM
        amp = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft_tmp], dtype="float32")
        amp = np.power(amp, 2)                                # Power処理
        bind_bin = None     # bin(ex,144)を束ねるための配列
        for b_itr in range(bin_num):
            wing = wingrange + int(b_itr/24)          # 1オクターブ上がるごとに、1ずつ増える
            target = freqs[b_itr] * overtone          # 抽出対象は倍音。引数でもらった分だけふえる 
            for search_idx in range(len(fft_freqs)):
                if target < fft_freqs[search_idx]:
                    weight = np.bartlett(wing + 1 + wing) # 荷重平均の計算- 重さを決定
                    cut = amp[search_idx - wing : search_idx + wing + 1] # 荷重平均をかける範囲の選択
                    avcut = np.average(cut, weights=weight)# カットして荷重平均をとったもの
                    bind_bin = np.copy(avcut) if bind_bin is None else np.hstack((bind_bin, avcut))  
                    # print("frame = {}, bin = {}, target = {}, fftbin = {} avcut = {} wing = {}".format(f_itr, b_itr, target, fft_freqs[search_idx], avcut, wing))
                    break
        bind_frame = np.copy(bind_bin) if bind_frame is None else np.vstack((bind_frame, bind_bin))
        center += CQTHOP
    bind_frame = bind_frame.reshape(1, bind_frame.shape[0], bind_frame.shape[1])
    return np.vstack((cqt, bind_frame))


if __name__ == "__main__":
    main()



"""
これより下はゴミ箱
早くgitの使い方を覚えような！
2019.12.10現在のexOvertone。これ少し削って精査する。ぞ

def exOvertone(cqt, wav, hop = CQTHOP, overtone=2, sr=SR, sample=SAMPLE_MUM, wingrange = 4, fmin=None, note_resolution = note_resolution, num_oct = 6):
    frame_num = cqt.shape[1]                                   # フレーム回数
    bin_num = cqt.shape[2]                                     # ループ回数
    freqs = GenMusicalScale(fmin, note_resolution, num_oct)    # 周波数瓶
    fft_freqs = np.fft.fftfreq(sample, 1/sr)
    zeropad = np.zeros(int(sample/2), dtype="float32")
    hsample = int(sample / 2)
    wav_pad = np.hstack((zeropad, wav, zeropad))
    center = hsample
    bind_frame = None       # 1フレームのbind_binをフレーム数分束ねたもの。これを元のCQT配列にvstackする 2019.12.11
    for f_ittr in tqdm(range(frame_num)):
        cut = wav_pad[center - hsample : center + hsample]
        wcut = cut * hammingWindow(len(cut))
        amp = np.abs(np.fft.fft(a=wcut) / (SAMPLE_MUM))
        bind_bin = None     # bin(ex,144)を束ねるための配列
        for b_ittr in range(bin_num):
            wing = wingrange + int(b_ittr/24)          # 1オクターブ上がるごとに、1ずつ増える
            target = freqs[b_ittr] * overtone          # 抽出対象は倍音。引数でもらった分だけふえる 
            # root = freqs[b]                     # CQTビンに含まれている周波数
            # overfreq = [root * i for i in range(2, overtone + 1)]   # rootを基音とした倍音 But! 今回の実装は2倍音まで 2019.12.10
            for search_idx in range(len(fft_freqs)):
                if target < fft_freqs[search_idx]:
                    weight = np.bartlett(wing + 1 + wing) # 荷重平均の計算- 重さを決定
                    cut = amp[search_idx - wing : search_idx + wing + 1] # 荷重平均をかける範囲の選択
                    avcut = np.average(cut, weights=weight)# カットして荷重平均をとったもの
                    bind_bin = np.copy(avcut) if bind_bin is None else np.hstack((bind_bin, avcut))  # なぜSyntax Error?
                    # print("frame = {}, bin = {}, target = {}, fftbin = {} avcut = {} wing = {}".format(f, b_ittr, target, fft_freqs[search_idx], avcut, wing))
                    break
        bind_frame = np.copy(bind_bin) if bind_frame is None else np.vstack((bind_frame, bind_bin))
        center += CQTHOP
    bind_frame = bind_frame.reshape(1, bind_frame.shape[0], bind_frame.shape[1])
    return np.vstack((cqt, bind_frame))
"""

