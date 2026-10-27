# E3SM validation data

`Reduced_land-model_output_lake-site_simulation.csv` is the compact validation
fixture retained in the repository.

The full hourly lake and land exports are intentionally kept out of normal Git
because they are larger than GitHub's 100 MB file limit:

- `Hourly lake-physics output from the E3SM lake-site simulation.csv`
- `Hourly land-model output from the E3SM lake-site simulation.csv`

Place those files in this folder locally before running the full comparison or
function-approximation workflows. They should be published through Git LFS or
an external data release when a complete reproduction archive is prepared.
