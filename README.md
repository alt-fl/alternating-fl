# Alt-FL

Todo...


## Build Alt-FL using Apptainer

An example specification file for Alt-FL is provided in `alt-fl.def`, use the
following command to build an `.sif` image:
```bash
apptainer build alt-fl.sif alt-fl.def 

```

Then run it with this command:
```bash
apptainer run --bind <PATH_TO_ALT_FL>:/app <PATH_TO_IMAGE> [ALT_FL_ARGUMENTS]
```

