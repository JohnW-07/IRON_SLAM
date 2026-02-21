# IRON_SLAM

# RUNNING_COMMANDS:

### Fast mode (2fps, good for 3-6 sec clips)
python pipeline.py --model snow.obj --video site.mp4 --outdir results/

### Accurate mode (every frame)
python pipeline.py --model snow.obj --video site.mp4 --outdir results/ --mode accurate

### Keep frames/depth maps for inspection
python pipeline.py --model snow.obj --video site.mp4 --outdir results/ --keep-frames