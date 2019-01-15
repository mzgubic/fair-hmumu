# fair-hmumu

# set up the first time

source the lcg view:
```
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3python3/latest/x86_64-centos7-gcc62-opt/setup.sh
```

set up the python virtual env
```
python3 -m venv fair-hmumu
cd fair-hmumu
pip install uproot
```

# set up every time you log in (CentOS machine)
```
source bin/activate
source setup_env.sh
```
