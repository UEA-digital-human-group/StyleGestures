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

def cut_bvh(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting BVH {filename} into intervals of {timespan}')
    
    p = BVHParser()
    bvh_data = p.parse(filename)
    if endtime<=0:
        endtime = bvh_data.framerate*bvh_data.values.shape[0]

    writer = BVHWriter()
    suffix=0
    while (starttime+timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        bvh_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.bvh')
        start_idx = int(np.round(starttime/bvh_data.framerate))
        end_idx = int(np.round((starttime+timespan)/bvh_data.framerate))+1
        if end_idx >= bvh_data.values.shape[0]:
            return
            
        with open(bvh_outfile,'w') as f:
            writer.write(bvh_data, f, start=start_idx, stop=end_idx)
            
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
        
def import_data(filename, motion_path, speech_data, style_path, id_path, ds_path, mirror=False, start=0, end=None):
    """Loads a file and concatenate all features to one [time, features] matrix. 
     NOTE: All sources will be truncated to the shortest length, i.e. we assume they
     are time synchronized and has the same start time."""
    
    motion_file = filename 
    id_file = filename
    if mirror:
        suffix="_mirrored"
        identity, basename = os.path.split(filename)
        identity = identity + suffix
        motion_file = os.path.join(identity, basename)
        
    motion_data = np.load(os.path.join(motion_path, motion_file + '.npy')).astype(np.float32)        
    n_motion_feats = motion_data.shape[1]

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

    if id_path is not None:
        if mirror:
            suffix="_mirrored"
            identity, basename = os.path.split(filename)
            identity = identity + suffix
            id_file = os.path.join(identity, basename)
        id_data = np.load(os.path.join(id_path, id_file + '.npy')).astype(np.float32)
        n_id_feats = id_data.shape[1]
        control_data = align(control_data,id_data)
    else:
        n_id_feats = 0

    if ds_path is not None:
        ds_data = np.load(os.path.join(ds_path, filename + '.npy')).astype(np.float32)
        n_ds_feats = ds_data.shape[1]
        control_data = align(control_data,ds_data)
    else:
        n_ds_feats = 0

    concat_data = align(motion_data, control_data)
    
    if np.isnan(concat_data).any():
        print('concat_data has nans')
        exit(0)
    if not end:
        end = concat_data.shape[0]
        
    return concat_data[start:end,:], n_motion_feats, n_speech_feats, n_style_feats, n_id_feats, n_ds_feats

def import_and_slice(files, motion_path, speech_data, style_path, id_path, ds_path, slice_window, slice_overlap, mirror=False, start=0, end=None):
    """Imports all features and slices them to samples with equal length time [samples, timesteps, features]."""
                    
    fi=0
    for file in files:
        print(file)
        
        # slice dataset
        concat_data, n_motion_feats, n_speech_feats, n_style_feats, n_id_feats, n_ds_feats = import_data(file, motion_path, speech_data, style_path, id_path, ds_path, False, start, end)        
        sliced = slice_data(concat_data, slice_window, slice_overlap)

        filenames = [file] * len(sliced)
        clipnos = range(len(sliced))

        if mirror:
            concat_mirr, nmf, n_speech_feats, n_style_feats, n_id_feats, n_ds_feats = import_data(file, motion_path, speech_data, style_path, id_path, ds_path, True, start, end)
            sliced_mirr = slice_data(concat_mirr, slice_window, slice_overlap)
            
            # append to the sliced dataset
            sliced = np.concatenate((sliced, sliced_mirr), axis=0)
        
        if fi==0:
            out_data = sliced
            out_filenames = filenames
            out_clipnos = clipnos
        else:
            out_data = np.concatenate((out_data, sliced), axis=0)
            out_filenames = np.concatenate((out_filenames, filenames), axis=0)
            out_clipnos = np.concatenate((out_clipnos, clipnos), axis=0)
        fi=fi+1

    return out_data[:,:,:n_motion_feats], out_data[:,:,n_motion_feats:], out_filenames, out_clipnos, n_speech_feats, n_style_feats, n_id_feats, n_ds_feats
    
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
    train_window_secs = 10
    test_window_secs = 60
    window_overlap = 0.8
    fps = 25

    # data_root = '../data/trinity/source'
    # bvhpath = '/data1/w0457094/data/udhopenpose3D/speakingsegments_upper_joint_imputed/sp1/'
    audiopath = '/data1/w0457094/data/udhaudio/segments_25kHz/'
    # speaker_ids = ['sp1', 'sp2', 'sp3']
    speaker_ids = ['sp1']
    # held_out = ['sp1/426G1404_03_000', 'sp1/426G1407_01_000', 'sp1/426G1408_02_000', 'sp2/426G1401_03_000']
    # held_out = ['sp1/426G1404_05_000', 'sp1/426G1405_02_000', 'sp1/426G1408_01_001']
    held_out = ['sp1/426G1404_05_000']
    processed_dir = '/data1/w0457094/data/processed_stylegestures/quat10s'    
    
    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(audiopath):
        for file in f:
            if '.wav' in file:
                ff=os.path.join(r, file)
                speakerid = os.path.split(r)[-1]
                if speakerid in speaker_ids:                
                    basename = os.path.splitext(os.path.basename(ff))[0]
                    # # IGNORE ACTED SEQUENCES
                    # if basename[0:8] == '426G1408' or basename[0:8] == '426G1409' or basename[0:8] == '426G1410':
                    #     continue
                    files.append(os.path.join(speakerid,basename))

    print(files)
    # motion_feat = 'joint_rot'
    speech_feat = 'melspec'
    
    # processed data will be organized as following
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    path = os.path.join(processed_dir, f'features_{fps}fps')
    motion_path = '/data1/w0457094/data/udhopenpose3D/segments_upper_joints_pca/'
    speech_path = os.path.join(path, f'{speech_feat}')
    style_path = '/data1/w0457094/data/speechcontent/'
    id_path = '/data1/w0457094/data/udhidentity/'
    # ds_path = '/data1/w0457094/data/udhaudio/16kHz/speakinglisteningsegmentsds/'
    # id_path = None
    ds_path = None


    if not os.path.exists(path):
        os.makedirs(path)
        
    # speech features
    if not os.path.exists(speech_path):
        print('Processing speech features...')
        os.makedirs(speech_path)
        extract_melspec(audiopath, files, speech_path, fps)
    else:
        print('Found speech features. skipping processing...')
    
    # # upper body joint angles
    # if not os.path.exists(motion_path):
    #     print('Processing motion features...')
    #     os.makedirs(motion_path)
    #     extract_joint_angles(bvhpath, files, motion_path, fps, fullbody=False)
    #     # full body joint angles
    #     #extract_joint_angles(bvhpath, files, motion_path, fps, fullbody=True)
    # else:
    #     print('Found motion features. skipping processing...')
    
    # # copy pipeline for converting motion features to bvh
    # copyfile(os.path.join(motion_path, 'data_pipe.sav'), os.path.join(processed_dir,f'data_pipe_{fps}fps.sav'))
   
    # # optional style features    
    # if not os.path.exists(hand_path):
    #     print('Processing style features...')
    #     os.makedirs(hand_path)
    #     os.makedirs(vel_path)
    #     os.makedirs(radius_path)
    #     os.makedirs(rh_path)
    #     #os.makedirs(lh_path)
    #     os.makedirs(sym_path)
    #     extract_hand_pos(bvhpath, files, hand_path, fps)
    #     extract_style_features(hand_path, files, path, fps, average_secs=4)
    # else:
    #     print('Found style features. skipping processing...')
        
    
    # divide in train, val, dev and test sets. Note that val and dev contains the same data allthough sliced in different ways.
    # - val data is used for logging and sliced the same way as the training data 
    # - dev data is sliced in longer sequences and used for visualization (we found that shorter snippets are hard to subjectivly evaluate)
    print("Preparing datasets...")
    
    train_files = [f for f in files if f not in held_out]
    
    slice_win_train = train_window_secs*fps
    slice_win_test = test_window_secs*fps
    val_test_split = 10*test_window_secs*fps # 10 
    
    train_motion, train_ctrl, train_fnames,train_clipnos, n_speech_feats, n_style_feats, n_id_feats, n_ds_feats = import_and_slice(train_files, motion_path, speech_path, style_path, id_path, ds_path, slice_win_train, window_overlap, mirror=True)
    val_motion, val_ctrl, val_fnames, val_clipnos,_,_,_,_ = import_and_slice(held_out, motion_path, speech_path, style_path, id_path, ds_path, slice_win_train, window_overlap, mirror=True, start=0, end=val_test_split)

    # the following sets are cut into longer clips without overlap. These are used for subjective evaluations during tuning (dev) and evaluation (test)
    dev_motion, dev_ctrl, dev_fnames,dev_clipnos,_,_,_,_ = import_and_slice(held_out, motion_path, speech_path, style_path, id_path, ds_path, slice_win_test, 0, mirror=False)

    # dev_motion, dev_ctrl, dev_fnames,dev_clipnos,_,_,_,_ = import_and_slice(held_out, motion_path, speech_path, style_path, id_path, ds_path, slice_win_test, 0, mirror=False, start=0, end=val_test_split)
    # test_motion, test_ctrl, test_fnames,test_clipnos,_,_,_,_ = import_and_slice(held_out, motion_path, speech_path, style_path, id_path, ds_path, slice_win_test, 0, mirror=False, start=val_test_split)
    

    print(train_ctrl.shape, val_ctrl.shape, dev_ctrl.shape)
    print(n_speech_feats, n_style_feats, n_id_feats, n_ds_feats)
    # # if style controlled, set the control values to 15%, 50% and 85% quantiles
    # if style_path is not None:
    #     dev_ctrl[0::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.15))
    #     dev_ctrl[1::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.5))
    #     dev_ctrl[2::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.85))
    #     test_ctrl[0::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.15))
    #     test_ctrl[1::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.5))
    #     test_ctrl[2::3,:,-1].fill(np.quantile(train_ctrl[:,:,-1],0.85))

    # split control signal to write out seperate components
    speech_idx = range(0,n_speech_feats)
    style_idx = range(n_speech_feats, n_speech_feats+n_style_feats)
    id_idx = range(n_speech_feats+n_style_feats, n_speech_feats+n_style_feats+n_id_feats)
    ds_idx = range(n_speech_feats+n_style_feats+n_id_feats, n_speech_feats+n_style_feats+n_id_feats+n_ds_feats)

    print(speech_idx,style_idx,id_idx,ds_idx)

    #import pdb;pdb.set_trace()
    np.savez(os.path.join(processed_dir,f'train_output_{fps}fps.npz'), clips = train_motion, fnames = train_fnames, clipnums = train_clipnos)
    np.savez(os.path.join(processed_dir,f'train_input_{fps}fps.npz'), clips = train_ctrl, fnames = train_fnames, clipnums = train_clipnos)
    np.savez(os.path.join(processed_dir,f'train_input_ds_{fps}fps.npz'), clips = train_ctrl[:,:,ds_idx], fnames = train_fnames, clipnums = train_clipnos)
    np.savez(os.path.join(processed_dir,f'val_output_{fps}fps.npz'), clips = val_motion, fnames = val_fnames, clipnums = val_clipnos)
    np.savez(os.path.join(processed_dir,f'val_input_{fps}fps.npz'), clips = val_ctrl, fnames = val_fnames, clipnums = val_clipnos)
    np.savez(os.path.join(processed_dir,f'val_input_ds_{fps}fps.npz'), clips = val_ctrl[:,:,ds_idx], fnames = val_fnames, clipnums = val_clipnos)
    np.savez(os.path.join(processed_dir,f'dev_output_{fps}fps.npz'), clips = dev_motion, fnames = dev_fnames, clipnums = dev_clipnos)
    np.savez(os.path.join(processed_dir,f'dev_input_{fps}fps.npz'), clips = dev_ctrl, fnames = dev_fnames, clipnums = dev_clipnos)
    np.savez(os.path.join(processed_dir,f'dev_input_ds_{fps}fps.npz'), clips = dev_ctrl[:,:,ds_idx], fnames = dev_fnames, clipnums = dev_clipnos)
    # np.savez(os.path.join(processed_dir,f'test_output_{fps}fps.npz'), clips = test_motion, fnames = test_fnames, clipnums = test_clipnos)
    # np.savez(os.path.join(processed_dir,f'test_input_{fps}fps.npz'), clips = test_ctrl, fnames = test_fnames, clipnums = test_clipnos)
    # np.savez(os.path.join(processed_dir,f'test_input_ds_{fps}fps.npz'), clips = test_ctrl[:,:,ds_idx], fnames = test_fnames, clipnums = test_clipnos)

    # finally prepare data for visualisation, i.e. the dev and test data in wav and bvh format    
    dev_vispath = os.path.join(processed_dir, 'visualization_dev')
    # test_vispath = os.path.join(processed_dir, 'visualization_test')    
    if not os.path.exists(dev_vispath):
        os.makedirs(dev_vispath)
        # os.makedirs(test_vispath)
        
        for file in held_out:
            cut_audio(os.path.join(audiopath, file + ".wav"), test_window_secs, dev_vispath, starttime=0.0,endtime=-1)

            # cut_audio(os.path.join(audiopath, file + ".wav"), test_window_secs, dev_vispath, starttime=0.0,endtime=10*test_window_secs)
            # cut_audio(os.path.join(audiopath, file + ".wav"), test_window_secs, test_vispath, starttime=10*test_window_secs)
    else:
        print('Found visualization data. Skipping processing...')
