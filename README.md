# Earth

```
cd ~

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# create conda env
conda env create -f CbrainCustomLayer.yml

# activate env 
conda activate CbrainCustomLayer

# get sherpa
git clone https://github.com/jordanott/sherpa.git
mv sherpa ~/miniconda3/envs/CbrainCustomLayer/lib/python3.6/site-packages/

# run search
python runner.py
```
