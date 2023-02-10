mount=--mount type=bind,source="$(CURDIR)/temp/report",target=/aster/work/tests/fem/temp \
      --mount type=bind,source="$(CURDIR)/temp/scratch",target=/aster/work/scratch

define check_and_create_dir
if [ ! -d temp/scratch ]; then \
    mkdir -p temp/scratch; \
fi
if [ ! -d temp/report ]; then \
    mkdir -p temp/report; \
fi
endef

dev:
	mamba env update --file environment.dev.yml --prune

format:
	black --config pyproject.toml . && isort . && ruff . --fix

bump:
	bumpversion patch setup.py

docs-dev:
	mamba env update --file docs/environment.docs.yml --prune

docs:
	activate adadocs && cd docs && make html

bbase:
	docker build . -t krande/ada:base -f images/base.Dockerfile

bdev:
	docker build . -t krande/ada:dev -f images/dev.Dockerfile

bfem:
	docker build . -t krande/ada:femtests -f images/femtests.Dockerfile

mdir:
	mkdir -p temp/report && mkdir temp/scratch

dtest:
	$(check_and_create_dir); \
 	docker run --rm $(mount) krande/ada:femtests bash -c "\
 		conda run --live-stream -n adadocker \
 		pytest . && \
 		python build_verification_report.py"

dprint:
	docker run --rm $(mount) krande/ada:femtests ls -l

dcheck:
	docker run --rm krande/ada:femtests ls -l /bin/bash

pbase:
	docker push krande/ada:base

run:
	docker run -it --rm -p 8888:8888 krande/adabase:latest

test:
	cd tests && pytest --cov=ada --cov-report=xml --cov-report=html .
