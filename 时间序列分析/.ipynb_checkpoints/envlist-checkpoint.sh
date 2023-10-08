
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=TS_ENV_38_and_AKshare

pip install akshare -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com  --upgrade
conda install pandas numpy scikit-learn matplotlib seaborn statsmodels
