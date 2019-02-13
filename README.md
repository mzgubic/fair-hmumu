# fair-hmumu

Train Hmumu classifiers adversarially to be fair w.r.t. the mass.


### set up the first time

```
git clone git@github.com:mzgubic/fair-hmumu.git
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3python3/latest/x86_64-centos7-gcc62-opt/setup.sh
python3 -m venv fair-hmumu
cd fair-hmumu
source bin/activate
pip install uproot
pip install tables
pip install root_pandas
pip install tensorflow_probability
```

### set up every time you log in (CentOS machine)
```
source setup_env.sh
```
