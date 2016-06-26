.PHONY: test

clean:
	python setup.py clean
	rm -f MANIFEST
	rm -rf dist
	rm -rf semlm.egg-info/
	rm -rf build
	# find . -name __pycache__ -exec rm -rf '{}' \;
	find . -name *.pyc -exec rm -rf '{}' \;

test:
	python3 -m unittest discover test

coverage:
	python3 -m coverage erase
	python3 -m unittest tests/test_main.py -v
	python3 -m coverage run tests/test_main.py
	python3 -m coverage report

gendoc:
	pydoc -w `find ../semlm -name '*.py'`

showdoc:
	pydoc ../semlm/*
