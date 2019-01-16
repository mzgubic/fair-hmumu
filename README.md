# fair-hmumu

Train Hmumu classifiers adversarially to be fair w.r.t. the mass.


### set up the first time

```
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3python3/latest/x86_64-centos7-gcc62-opt/setup.sh
python3 -m venv fair-hmumu
cd fair-hmumu
pip install uproot
pip install tables
pip install root_pandas
```

### set up every time you log in (CentOS machine)
```
source bin/activate
source setup_env.sh
```
