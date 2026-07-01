# How to use

Use the converter at https://meraak1.github.io/chunithmtodancerail3/

Alternatively, download the ugc2dr3.py script and run it:

```
py ugc2dr3.py song1.zip song2.zip
```

Optional arguments:

```
-o, --o directory for the output zips (default: next to each input)
--name file DR3 keyword (single-file use only; from title by default)
--level DR3 tier number (difficulty)
--ln-density LN centre spacing as a beat fraction (this adds center notes to LNs so you have to hold them, but ignores LNs that already have more center holds than what would be added, default 1/4)
--flick-tap overlay a type 1 tap on every flick (default off)
--no-head-tap do NOT overlay strict taps on LN heads (they are overlayed by default)
```
