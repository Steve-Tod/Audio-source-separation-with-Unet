import numpy as np 
import wave
import os
import struct
import nussl

def audio_decomp(video=None,audio=None,OutputPath=None,file=None):
    signal=nussl.AudioSignal(audio_data_array=audio[0,:],sample_rate=44100)
    nmf_mfcc=nussl.NMF_MFCC(signal,num_sources=2, num_templates=25, distance_measure="euclidean", random_seed=0)
    nmf_mfcc.run()
    sources=nmf_mfcc.make_audio_signals()
    source_list=[]
    for i,source in enumerate(sources):
        outputfile=os.path.join(OutputPath,file+'_seg'+str(i+1)+'.wav')
        source.write_audio_to_file(outputfile)
        source_list.append(source.audio_data)

    return [os.path.join(OutputPath,file+"_seg1.wav"),os.path.join(OutputPath,file+"_seg2.wav")],source_list

def main():
    OutputPath='result_audio'
    namelist=os.listdir('testimage')
    for name in namelist:
        signal1=nussl.AudioSignal(path_to_input_file='gt_audio/'+name+'_gt1.wav')
        signal2=nussl.AudioSignal(path_to_input_file='gt_audio/'+name+'_gt2.wav')
        signal=nussl.AudioSignal(path_to_input_file='gt_audio/'+name+'.wav')
        audio=signal.audio_data
        m=audio_decomp(audio=audio,OutputPath=OutputPath,file=name)
        signal3=nussl.AudioSignal(path_to_input_file='result_audio/'+name+'_seg1.wav')
        signal4=nussl.AudioSignal(path_to_input_file='result_audio/'+name+'_seg2.wav')
        ref_sources=np.zeros([2,len(audio[0,:])])
        est_sources=np.zeros([2,len(audio[0,:])])
        ref_sources[0,:]=sum(signal1.audio_data)
        ref_sources[1,:]=sum(signal2.audio_data)
        est_sources[0,:]=sum(signal3.audio_data)
        est_sources[1,:]=sum(signal4.audio_data)
        result=separation.bss_eval_sources(ref_sources,est_sources,compute_permutation=True)
        print result[0]

if __name__=='__main__':
    main()








