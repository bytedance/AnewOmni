## Generation

To generate Top-$K$ samples ranked by confidence, which is the lower the better as it predicts the atomic pairwise distance error):

```bash
python -m api.generate_with_rank --config api/demo/config_rank.yaml --confidence_ckpt /path/to/confidence_model.ckpt --save_dir /path/to/save/results
```

If you want to rank by likelihood, please add the option `--rank_criterion likelihood`.

The generation and filtering process will continue until the top-$K$ samples are stable for at least $N$ more samples checked, where $N$ is specified as `patience` in the config file. Usually a larger patience value will lead to more trials and longer running time before accomplishment.

If you just want to generate $K$ samples without the top-$K$ stability check, you can specify `patience` to zero, in which case the process will terminate instantly when the number of saved candiates reaches $K$. It is further recommended to set `--rank_criterion none` in this case to disable sorting, as it is more efficient to do it after all candidates are generated.


## Filters

The framework supports custom implementation of filters. Only samples passing through all filters will be put into the queue for the ranking process mentioned above. Please refer to `filters/chem.py` for a simple example of a molecular weight filter. To get properties from the structural data, you can load results from PDB/mmCIF/SDF in the forward call of filters by appending suitable extensions to the `path_prefix` in `FilterInput` (e.g. `pdb_path = input.path_prefix + '.pdb'`).

Some reminders:
- The decorator `@R.register('ClassName')` right before your filter class to let the framework locate your definitions.
- The decorator `@ray.remote(num_cpus=x, num_gpus=x)` to let the ray framework know the resources required for a single call of this filter.

## Confidence Visualization

To visualize the heatmap of a single candidate, please use the following command:

```bash
python api/visual_confidence/run.py /path/to/candidate.pdb
```

Then you can see the results from `localhost:8000`. Note that vscode will automatically forward the address to your local machine, but you can also use the preview function of vscode to directly see the webpage in the editor.

The coloring of atoms are based on lig/cplx_PDE on 6-angstrom neighborhood.

## Resample Mode

Sampling under the local neighborhood of a given start point: just use the template `MoleculeResample` is ok.

## Structure Prediction

You only need to adjust the config file to do structure prediction.

example:

```bash
python -m api.generate_with_rank --config api/demo/config_structpred.yaml --confidence_ckpt /path/to/confidence_model.ckpt --save_dir /path/to/save/results
```

In the config file, you can specify one single molecule for structure prediction:

```yaml
- class: StructPredMolecule
  candidate_name: 6IC
  seq: C#Cc1c(ccc2c1c(cc(c2)O)c3c(c4c(cn3)c(nc(n4)OC[C@@]56CCCN5C[C@@H](C6)F)N7C[C@H]8CC[C@@H](C7)N8)F)F
  w: 2.0
```

You can also specify a batch of sequences in a txt file with the format `<name> <sequence>` (splitted by space of \t).

```yaml
- class: StructPredTemplateFactory  # example of batch configuration
  config:
    class: StructPredLinearPeptide
    w: 1.0
  file_path: ./api/demo/data/pep.txt
```

The test case on the MIRATI molecule on KRAS 12D (7RPZ) shows that when confidence < 1, there might be some reliable predictions.

Please refer to the example configuration `api/demo/config_structpred.yaml` for more details.