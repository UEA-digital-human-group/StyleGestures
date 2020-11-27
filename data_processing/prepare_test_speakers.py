import numpy as np

import glob
import os
import sys
from shutil import copyfile
from audio_features import extract_melspec
import scipy.io.wavfile as wav
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *

def cut_audio(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting AUDIO {filename} into intervals of {timespan}')
    fs,X = wav.read(filename)
    if endtime<=0:
        endtime = len(X)/fs
    suffix=0
    while (starttime+timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        wav_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.wav')
        start_idx = int(np.round(starttime*fs))
        end_idx = int(np.round((starttime+timespan)*fs))+1
        if end_idx >= X.shape[0]:
            return
        wav.write(wav_outfile, fs, X[start_idx:end_idx])
        starttime += timespan
        suffix+=1
        
def slice_data(data, window_size, overlap):

    nframes = data.shape[0]
    overlap_frames = (int)(overlap*window_size)
    
    n_sequences = (nframes-overlap_frames)//(window_size-overlap_frames)
    
    if n_sequences>0:
        sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)

        # extract sequences from the data
        for i in range(0,n_sequences):
            frameIdx = (window_size-overlap_frames) * i
            sliced[i,:,:] = data[frameIdx:frameIdx+window_size,:].copy()
    else:
        print("WARNING: data too small for window")
        sliced = np.zeros((0, window_size, data.shape[1])).astype(np.float32)
                    
    return sliced
    
def align(data1, data2):
    """Truncates to the shortest length and concatenates"""
    
    nframes1 = data1.shape[0]
    nframes2 = data2.shape[0]
    if nframes1<nframes2:
        return np.concatenate((data1, data2[:nframes1,:]), axis=1)
    else:
        return np.concatenate((data1[:nframes2,:], data2), axis=1)
        
def import_data(filename, speech_data, style_path, id_label=None):
    """Loads an audio file and concatenate style data"""

    basename = os.path.split(filename)[-1]
    speech_data = np.load(os.path.join(speech_path, basename + '.npy')).astype(np.float32)
    n_speech_feats = speech_data.shape[1]

    control_data = speech_data
    
    if style_path is not None:
        style_data = np.load(os.path.join(style_path, filename + '.npy')).astype(np.float32)
        n_style_feats = style_data.shape[1]
        control_data = align(control_data,style_data)
    else:
        n_style_feats = 0
    
    if id_label is not None:
        id_data = np.repeat(id_label,control_data.shape[0], axis= 0)
        control_data = align(control_data,id_data)
        n_id_feats = len(id_label)
    else:
        n_id_feats = 0

    return control_data, n_speech_feats, n_style_feats, n_id_feats

def import_and_slice(files, speech_data, style_path, id_label, slice_window):
    """Imports all features and slices them to samples with equal length time [samples, timesteps, features].
    """                 
    fi=0
    for file in files:
        print(file)
        
        # slice dataset
        concat_data, n_speech_feats, n_style_feats, n_id_feats = import_data(file, speech_data, style_path, id_label)        
        sliced = slice_data(concat_data, slice_window, overlap=0)

        filenames = [file] * len(sliced)
        clipnos = range(len(sliced))
        
        if fi==0:
            out_data = sliced
            out_filenames = filenames
            out_clipnos = clipnos
        else:
            out_data = np.concatenate((out_data, sliced), axis=0)
            out_filenames = np.concatenate((out_filenames, filenames), axis=0)
            out_clipnos = np.concatenate((out_clipnos, clipnos), axis=0)
        fi=fi+1

    return out_data, out_filenames, out_clipnos, n_speech_feats, n_style_feats, n_id_feats
    
if __name__ == "__main__":
    '''
    Converts motion and wav files into features, slices in equal length intervals and divides the data
    into training, validation and test sets. 
    
    Adding an optional style argument ("MG-R", "MG-V", "MG-H" or "MS-S") 
    will add features for style control.
    '''     
    if len(sys.argv)==1:
        style_path = None
    elif len(sys.argv)==2:
        style_path = sys.argv[1]
    else:
        print("usage: python prepare_datasets.py [MS-S|MG-R|MG-V|MG-H]")
        sys.exit(0)
     
    # Hardcoded preprocessing params and file structure. 
    # Modify these if you want the data in some different format
    test_window_secs = 60
    window_overlap = 0.8
    fps = 25

    audiopath = '/data1/w0457094/data/other_speech/audio_25kHz/'
    processed_dir = '/data1/w0457094/data/processed_stylegestures/other_speech'    
    num_identities = 2

    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(audiopath):
        for file in f:
            if '.wav' in file:
                ff=os.path.join(r, file)
            
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    print(files)
    speech_feat = 'melspec'
    
    # processed data will be organized as following
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    path = os.path.join(processed_dir, f'features_{fps}fps')
    speech_path = os.path.join(path, f'{speech_feat}')
    style_path = '/data1/w0457094/data/other_speech/speechcontent'


    if not os.path.exists(path):
        os.makedirs(path)
        
    # speech features
    if not os.path.exists(speech_path):
        print('Processing speech features...')
        os.makedirs(speech_path)
        extract_melspec(audiopath, files, speech_path, fps)
    else:
        print('Found speech features. skipping processing...')

    
    # divide in train, val, dev and test sets. Note that val and dev contains the same data allthough sliced in different ways.
    # - val data is used for logging and sliced the same way as the training data 
    # - dev data is sliced in longer sequences and used for visualization (we found that shorter snippets are hard to subjectivly evaluate)
    print("Preparing datasets...")
        
    slice_win_test = test_window_secs*fps
    

    # cut into clips without overlap.
    for k in range(num_identities):
        id_label = np.zeros((1,num_identities))
        id_label[0,k] = 1
        test_ctrl, test_fnames, test_clipnos,_,_,_ = import_and_slice(files, speech_path, style_path, id_label,slice_win_test)

        np.savez(os.path.join(processed_dir,f'test_input_{fps}fps_id{k}.npz'), clips = test_ctrl, fnames = test_fnames, clipnums = test_clipnos)

    # finally prepare audio for visualisation    
    dev_vispath = os.path.join(processed_dir, 'visualization_test')
    if not os.path.exists(dev_vispath):
        os.makedirs(dev_vispath)
        
        for file in files:
            cut_audio(os.path.join(audiopath, file + ".wav"), test_window_secs, dev_vispath, starttime=0.0,endtime=-1)
    else:
        print('Found visualization data. Skipping processing...')
