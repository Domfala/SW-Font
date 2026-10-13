# Dataset Splits

The dataset comprises 500 Chinese font styles, divided into 440 seen styles (SF) and 60 unseen styles (UF). At the character level, 800 Chinese characters are designated as seen characters (SC), and another 100 as unseen characters (UC).

The training set uses seen styles and seen characters (SFSC). Three evaluation settings are defined:

| Setting | Font Styles | Characters |
|---|---|---|
| SFUC | 30 seen styles | 100 unseen characters |
| UFSC | 30 unseen styles | 100 seen characters |
| UFUC | 30 unseen styles | 100 unseen characters |

Characters are further divided into three complexity levels according to stroke count:

- **Easy:** 5–9 strokes
- **Medium:** 10–20 strokes
- **Hard:** 21 or more strokes

Pixel-level stroke-width maps are constructed from the font images using A-SWT to support WES-Net pretraining and stroke-width supervision.

The original font data are not redistributed due to confidentiality and third-party licensing restrictions. Users should prepare appropriately licensed font data following the directory structure described in the README.
