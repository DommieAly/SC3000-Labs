# SC3000 Labs

Team NanoGangstas's work for SC3000 Lab 1 (teaching NanoGPT arithmetic with DPO) and Lab 2 (Prolog).

The repository holds several copies of some files, so this README records exactly which ones go to the TA.

## What to submit

Everything for the TA is in `submissions/`. Send only these files.

| Assignment | Send | To | Due |
|---|---|---|---|
| Lab 1 | `NanoGangstas.zip`, made by zipping the `submissions/Lab1/dpo/` folder | Lab-group TA, by email | 23:59, Fri 6 Nov 2026 |
| Lab 2 | The 4 files in `submissions/Lab2/` | TA, by email | 6 Nov 2026 (the manual gives no time) |

These files were copied on 6 Nov 2026 from commit `bc69729` on `main`, except the Lab 2 PDF, which was rebuilt after that commit from `Lab2/helpers/NanoGangstas_Assignment2.html`.

## Lab 1: `submissions/Lab1/dpo/`

| File | Copied from | Bytes | SHA-256 |
|---|---|---|---|
| `dpo.pt` | `Lab1/NanoGPT-Math/dpo/dpo.pt` | 35,370,101 | `f553db9f20f5821a72ef2300b35e80903749bac159c10a23e7868951a2be674d` |
| `dpo.ipynb` | `Lab1/NanoGPT-Math/dpo/dpo.ipynb` | 143,834 | `78b3900cc220f2ca113f9f9b32e90f59bb6a5f4b9ea96454b552c19504313e57` |
| `pos_neg_pairs.json` | `Lab1/NanoGPT-Math/dpo/pos_neg_pairs.json` | 13,972,972 | `8fc97f503762258e4e0453784e341c0e8b0ea932bf00a0e57af65941cb5ed85f` |

These files are not zipped yet. Compress the `dpo` folder and rename the resulting `dpo.zip` to `NanoGangstas.zip`, so the paths inside it start with `dpo/`.

## Lab 2: `submissions/Lab2/`

| File | Copied from | Bytes | SHA-256 |
|---|---|---|---|
| `NanoGangstas_Assignment2.pdf` | `Lab2/NanoGangstas_Assignment2.pdf` | 397,676 | `f89bf13be6543ec691b2ba983b5159a8d9eb02f14d059de2bd57148ef33bff67` |
| `NanoGangstas_qn_1_2.pl` | `Lab2/NanoGangstas_qn_1_2.pl` | 479 | `7770903a64e6be09df131ec780c3cb4233798a909b30fece9e8a956fdc108b65` |
| `NanoGangstas_qn_2_1.pl` | `Lab2/NanoGangstas_qn_2_1.pl` | 393 | `5125b26182b82fc2359d7d3e98127f41d524056c28b4264a7474c7b238a8dc0b` |
| `NanoGangstas_qn_2_2.pl` | `Lab2/NanoGangstas_qn_2_2.pl` | 571 | `12bd2f3d05105758019119536a7ddccea684429180518311261943c4044e5b07` |

The Lab 2 manual asks for `<team_name>_Assignment3.pdf`. We named the PDF `Assignment2` because this is Assignment 2.

## Do not submit

| Path | What it is | Why not |
|---|---|---|
| `Lab1/NanoGPT-Math/dpo_v1/` | Our first attempt (82.3% accuracy) | Older model. Its `dpo.pt` hash starts `844e6c19`; the final one starts `f553db9f` |
| `Lab1/NanoGPT-Math/dpo/` | Originals of the Lab 1 files | Same content as `submissions/Lab1/dpo/` |
| `Lab1/NanoGPT-Math/dpo_reasoning/` | Leftover Jupyter checkpoint, if it exists on your machine | Old folder name; git ignores it |
| `Lab1/pilots/` and `Lab1/check_pos_neg_pairs.py` | Experiment scripts and an older notebook | Working files only |
| `Lab2/*.pl` and `Lab2/*.pdf` | Originals of the Lab 2 files | Same content as `submissions/Lab2/` |
| `Lab2/helpers/` | Report source (the HTML and the earlier LaTeX version) and saved traces | Working files only |

## Check the files

Compare the output of this command with the tables above.

```bash
shasum -a 256 submissions/Lab1/dpo/* submissions/Lab2/*
```

After zipping, check the zip the same way. It should hold the 3 files under `dpo/`, with the same hashes as the table. Finder's Compress also adds a `__MACOSX/` metadata folder.

```bash
unzip -l NanoGangstas.zip
for f in dpo.pt dpo.ipynb pos_neg_pairs.json; do
  unzip -p NanoGangstas.zip "dpo/$f" | shasum -a 256
done
```

## If a source file changes

The files in `submissions/` are copies, so they do not pick up later edits. If a source file changes after commit `bc69729`, for example the notebook, copy it again from the repository root, update the hashes above, and zip Lab 1 again:

```bash
cp -p Lab1/NanoGPT-Math/dpo/dpo.pt Lab1/NanoGPT-Math/dpo/dpo.ipynb Lab1/NanoGPT-Math/dpo/pos_neg_pairs.json submissions/Lab1/dpo/
cp -p Lab2/NanoGangstas_qn_*.pl Lab2/NanoGangstas_Assignment2.pdf submissions/Lab2/
```

The Lab 2 PDF is printed from `Lab2/helpers/NanoGangstas_Assignment2.html`. The command to rebuild it is at the top of that file.
