# UMIGURI To DanceRail3

This is a converter that will convert UMIGURI beatmaps to DanceRail3 charts. Only UMIGURI beatmaps (.ugc) are supported.

This converter focuses on making the converted file fun to play in DR3, rather than going for 1:1 accuracy. Certain quirks like impossible bomb notes or bugged notes that would crash the game are automatically detected and removed. The converter outputs zip files that are ready to be imported into my DR3 Custom app for Android (https://www.youtube.com/watch?v=JSuXkqWYa4o) with no extra work required.

To find UMIGURI beatmaps, check https://pgko.dev and the UMIGURI Discord server (https://discord.com/invite/j6mEU2hQDZ)

This is designed to be used with custom community-made URIGUMI charts, NOT with official Chunithm charts.


# How to use

Use the converter at https://meraak1.github.io/chunithmtodancerail3/

Alternatively, download the ugc2dr3.py script and run it. (You need ffmpeg in PATH)

```
py ugc2dr3.py song1.zip song2.zip
```
