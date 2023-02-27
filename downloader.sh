# virtual-env
python3 -m venv .venv_facerec
source .venv_facerec/bin/activate
pip install -r requirements.txt

# download data
gdown --no-check-certificate https://drive.google.com/uc?id=1_t0yLXA-kU2vmUehfpN-Lv-jTAzjYuZG -O data/data.zip

# download unzip
sudo apt install unzip

# unzip data
unzip data/data.zip -d data/data

# rezmove .zip
rm data/data.zip
