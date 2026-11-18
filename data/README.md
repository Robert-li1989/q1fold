# Data

The full benchmark datasets are not stored in this repository.

| Dataset | Source | Split used | Sequences (≤ 512 nt) |
|---|---|---|---|
| bpRNA-1m | Danaee et al. (2018); TR0/VL0/TS0 partition from SPOT-RNA (Singh et al., 2019) | TR0 / VL0 / TS0 | 10,814 / 1,300 / 1,305 |
| RNAStrAlign | Tan et al. (2017); split of Chen et al. (2020, E2Efold) | train / valid / test | 15,988 / 1,977 / 2,001 |
| ArchiveII | Sloma & Mathews (2016) | test only | 3,380 |

The exact CSV files used in the paper (`TR0.csv`, `VL0.csv`, `TS0.csv`,
`train_512.csv`, `valid_512.csv`, `test_512.csv`, `ArchiveII_512.csv`) are
archived at: **[Zenodo DOI to be added]**

Place them as:

```
data/bpRNA-1m/{TR0,VL0,TS0}.csv
data/RNAstralign/{train,valid,test}_512.csv
data/RNAstralign/ArchiveII_512.csv
```

Each CSV has the columns `id`, `sequence` and `dotbracket`. Sequences longer
than 512 nucleotides were removed.

`data/sample/` holds a few sequences for a quick installation test.
