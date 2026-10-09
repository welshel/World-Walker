import json

def load_trajectory(path,expected_frames=61):
    d=json.load(open(path)); frames=d.get('frames',d if isinstance(d,list) else None)
    if not isinstance(frames,list) or len(frames)!=expected_frames: raise ValueError(f'expected {expected_frames} camera frames')
    indices=[f.get('frame',i) for i,f in enumerate(frames)]
    if indices!=list(range(expected_frames)): raise ValueError('camera frame indices are not native 000..060')
    return d
