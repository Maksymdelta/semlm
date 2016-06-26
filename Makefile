.PHONY: test

clean:
	python setup.py clean
	rm -f MANIFEST
	rm -rf dist
	rm -rf semlm.egg-info/
	rm -rf build
	find . -name *.pyc -exec rm -rf '{}' \;
	rm -rf htmlcov

test:
	python3 -m unittest discover test -v

coverage:
	python3 -m coverage erase
	python3 -m coverage run setup.py test
	python3 -m coverage html
	python3 -m coverage report

gendoc:
	pydoc -w `find ../semlm -name '*.py'`

showdoc:
	pydoc ../semlm/*
