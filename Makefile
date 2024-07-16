install:
	pip install .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	pytest tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files

documenation:
	# sphinx-apidoc -o docs dolomite_engine
	cd docs && make html && cd ..
