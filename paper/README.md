# Paper

English | [繁體中文](README.zh-TW.md)

Fu-Kai Chuang, *Estimating a shoebox room from a single room impulse response with unknown playback latency: simulated performance and measured failure of image-source lattice inversion*.

| File | Language |
| --- | --- |
| `paper.zh-TW.pdf` | Traditional Chinese, the version I wrote first |
| `paper.en.pdf` | English, the same paper rendered section by section |

The paper evaluates the lattice inversion of this repository (`inversion.method: lattice`) on simulated rooms and on the measured responses of dEchorate. It fails on the measured responses, and section 5.7 and appendix E diagnose why. The scripts of every experiment are in [`../experiments/`](../experiments/).

The paper is licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>). The code of the repository is under the MIT licence in [`../LICENSE`](../LICENSE).
