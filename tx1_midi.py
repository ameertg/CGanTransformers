from collections import defaultdict
import tempfile
import pickle

def midi_to_tx1(midi):
  import pretty_midi
  pretty_midi.pretty_midi.MAX_TICK = 1e16

#  # Load MIDI file
#  with tempfile.NamedTemporaryFile('wb') as mf:
#    #pickle.dump(midi,mf)
#    mf.write(midi)
#    mf.seek(0)
#    midi = pretty_midi.PrettyMIDI(mf.name)

  midi = pretty_midi.PrettyMIDI(midi)

  ins_names = ['p1', 'p2', 'tr', 'no']
  instruments = sorted(midi.instruments, key=lambda x: ins_names.index(x.name))
  samp_to_events = defaultdict(list)
  for ins in instruments:
    instag = ins.name.upper()

    last_start = -1
    last_end = -1
    last_pitch = -1
    for note in ins.notes:
      start = (note.start * 44100) + 1e-6
      end = (note.end * 44100) + 1e-6

      assert start - int(start) < 1e-3
      assert end - int(end) < 1e-3

      start = int(start)
      end = int(end)

      assert start > last_start
      assert start >= last_end

      pitch = note.pitch

      if last_end >= 0 and last_end != start:
        samp_to_events[last_end].append('{}_NOTEOFF'.format(instag))
      samp_to_events[start].append('{}_NOTEON_{}'.format(instag, pitch))

      last_start = start
      last_end = end
      last_pitch = pitch

    if last_pitch != -1:
      samp_to_events[last_end].append('{}_NOTEOFF'.format(instag))

  tx1 = []
  last_samp = 0
  for samp, events in sorted(samp_to_events.items(), key=lambda x: x[0]):
    wt = samp - last_samp
    assert last_samp == 0 or wt > 0
    if wt > 0:
      tx1.append('WT_{}'.format(wt))
    tx1.extend(events)
    last_samp = samp

  nsamps = int((midi.time_signature_changes[-1].time * 44100) + 1e-6)
  if nsamps > last_samp:
    tx1.append('WT_{}'.format(nsamps - last_samp))

  tx1 = '\n'.join(tx1)
  return tx1


def tx1_to_midi(tx1):
  import pretty_midi

  tx1 = tx1.strip().splitlines()
  nsamps = sum([int(x.split('_')[1]) for x in tx1 if x[:2] == 'WT'])

  # Create MIDI instruments
  p1_prog = pretty_midi.instrument_name_to_program('Lead 1 (square)')
  p2_prog = pretty_midi.instrument_name_to_program('Lead 2 (sawtooth)')
  tr_prog = pretty_midi.instrument_name_to_program('Synth Bass 1')
  no_prog = pretty_midi.instrument_name_to_program('Breath Noise')
  p1 = pretty_midi.Instrument(program=p1_prog, name='p1', is_drum=False)
  p2 = pretty_midi.Instrument(program=p2_prog, name='p2', is_drum=False)
  tr = pretty_midi.Instrument(program=tr_prog, name='tr', is_drum=False)
  no = pretty_midi.Instrument(program=no_prog, name='no', is_drum=True)

  name_to_ins = {'P1': p1, 'P2': p2, 'TR': tr, 'NO': no}
  name_to_pitch = {'P1': None, 'P2': None, 'TR': None, 'NO': None}
  name_to_start = {'P1': None, 'P2': None, 'TR': None, 'NO': None}
  name_to_max_velocity = {'P1': 15, 'P2': 15, 'TR': 1, 'NO': 15}

  samp = 0
  for event in tx1:
    if event == '<eos>':
        continue
    if event[:2] == 'WT':
      samp += int(event[3:])
    else:
      tokens = event.split('_')
      name = tokens[0]
      ins = name_to_ins[tokens[0]]

      old_pitch = name_to_pitch[name]
      if tokens[1] == 'NOTEON':
        if old_pitch is not None:
          ins.notes.append(pretty_midi.Note(
              velocity=name_to_max_velocity[name],
              pitch=old_pitch,
              start=name_to_start[name] / 44100.,
              end=samp / 44100.))
        name_to_pitch[name] = int(tokens[2])
        name_to_start[name] = samp
      else:
        if old_pitch is not None:
          ins.notes.append(pretty_midi.Note(
              velocity=name_to_max_velocity[name],
              pitch=name_to_pitch[name],
              start=name_to_start[name] / 44100.,
              end=samp / 44100.))

        name_to_pitch[name] = None
        name_to_start[name] = None

  # Deactivating this for generated files
  #for name, pitch in name_to_pitch.items():
  #  assert pitch is None

  # Create MIDI and add instruments
  midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=22050)
  midi.instruments.extend([p1, p2, tr, no])

  # Create indicator for end of song
  eos = pretty_midi.TimeSignature(1, 1, nsamps / 44100.)
  midi.time_signature_changes.append(eos)

#  with tempfile.NamedTemporaryFile('rb') as mf:
#    midi.write(mf.name)
#    midi = mf.read()

  return midi

def oneHot_TX1(one_hot):
    """
    Takes one-hot outputs of the model and returns a TX1 representation.

    @param one_hot: tensor of shape (sequence length, batch size, 631)

    @return outputs list of strings with the tx1 representation of each batch
    """
    argm = np.argmax(one_hot.detach().numpy(),axis=2)
    batch_size = argm.shape[1]
    outputs = []
    for b in range(batch_size):
        toks = [idx2tok[int(i)] for i in argm[:,b]]
        outputs.append("\n".join(toks))
    return outputs

#---USE THIS MAIN FOR MIDI TO TX1---
#if __name__ == "__main__":
#    import os
#    import pretty_midi
#    import pickle
#    in_dir = '5322LakhPopRockAdapted'
#    out_dir = '5322LakhPopRockTX1'
#    for i,fname in enumerate(os.listdir(in_dir)):
#        print(i)
#        #midi = pretty_midi.PrettyMIDI(os.path.join(in_dir,fname))
#        tx1_representation = midi_to_tx1(os.path.join(in_dir,fname))
#        with open(os.path.join(out_dir,fname.split('.')[0]) + '.txt','w') as f:
#            f.write(tx1_representation)

#---USE THIS MAIN FOR TX1 TO MIDI---
if __name__ == "__main__":
    import os
    import pretty_midi
    import pickle
    import torch
    import numpy as np

    with open('tx1_vocab.txt','r') as f: #make a dict mapping indices to TX1 tokens
        tokens = f.readlines()
        tokens = [tok.strip('\n') for tok in tokens]
        idx2tok = {i:tokens[i-1] for i in range(1,631)}
    idx2tok[0] = '<eos>'


    in_file = '../transformer_outs.p' #input file with list of tensors
    out_dir = 'transformer_outputs' #ouput directory to write midi batches to
    with open(in_file, 'rb') as f:
        tensor_list = pickle.load(f)

    for i,tensor in enumerate(tensor_list):
        tx1_representations = oneHot_TX1(tensor)
        for j,rep in enumerate(tx1_representations):
            midi = tx1_to_midi(rep)
            midi.write(os.path.join(out_dir,str(i) + "_" + str(j) + '.mid'))



    # in_dir = 'test_tx1_mid_in' #in_dir should have pytorch one hot tensors
    # out_dir = 'test_tx1_mid_out'
    # for i,fname in enumerate(os.listdir(in_dir)):
    #     tensor = torch.load(os.path.join(in_dir,fname))
    #     tx1_representations = oneHot_TX1(tensor)
    #     for j,rep in enumerate(tx1_representations):
    #         midi = tx1_to_midi(rep)
    #         midi.write(os.path.join(out_dir,fname.split('.')[0]) + str(j) + '.mid')
