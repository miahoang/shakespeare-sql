To use BIRD in the framework, please download [the BIRD dev, BIRD train, and BIRD minidev datasets](https://bird-bench.github.io/), then unzip the database files into this folder. The structure needs to look like this:

```
shakespear-sql/
├── data/                    
│   └── BIRD/                    
│       ├── bird_dev_ambrosia_format.csv
│       ├── bird_minidev_ambrosia_format.csv
│       ├── bird_train_ambrosia_format.csv
│       ├── database/
│            ├── dev/
│                 ├── california_schools/
│                 ├── card_games/
│                 ├── ...
│            ├── minidev/
│                 ├── california_schools/
│                 ├── card_games/
│                 ├── ...
│            ├── train/
│                 ├── address/
│                 ├── airlines/
│                 ├── ...
(the rest of the BIRD files are not needed)
```
