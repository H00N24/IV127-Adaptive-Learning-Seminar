# IV127-Adaptive-Learning-Seminar

IV127 Adaptive Learning Seminar

### Requirements

```
$ python3 -m venv venv
$ source venv/bin/activate
[venv] $ pip install --upgrade pip wheel setuptools
[venv] $ pip install -r requierements.txt
```

### Prepare data

```
[venv] $ python create_data.py
```

### Notebooks

```
[venv] $ python -m ipykernel install --user --name iv127 --display-name "IV127 Adaptive Learning Seminar"
[venv] $ jupyter notebook
```

### Run streamlit app

```
[venv] $ streamlit run app.py
```
