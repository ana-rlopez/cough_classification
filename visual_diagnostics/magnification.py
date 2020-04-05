import eulerian_magnification as em
import numpy as np
import skvideo.io

source_path = 'outpy.avi'
vid, fps = em.load_video_float(source_path)
result = em.eulerian_magnification(vid, fps,
        freq_min=0.8,
        freq_max=1,
        amplification=30,
        pyramid_levels=5
)

# initialize video writer
writer = skvideo.io.FFmpegWriter("result.mp4")

# new frame after each addition of water
for i in range(result.shape[0]):
    to_write = result[i, ...]
    to_write = to_write - to_write.min()
    to_write = to_write * 255
    to_write = to_write.astype(np.uint8)
    to_write = to_write[...,::-1]
    writer.writeFrame(to_write)
