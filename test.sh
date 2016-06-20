
python3 -m coverage erase
python3 -m unittest tests/test_main.py -v
python3 -m coverage run tests/test_main.py
python3 -m coverage report
