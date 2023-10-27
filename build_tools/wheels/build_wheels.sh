#!/bin/bash

set -e
set -x
echo "build_wheels.sh: running"

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

pythonversion=$(python --version | awk -e '{print $2}' | awk -F. -e '{print $1 $2}')
if [[ "${pythonversion}" == "38" ]]; then
  python -m pip install SimpleITK-SimpleElastix==2.0.0rc2.dev909
fi
python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
