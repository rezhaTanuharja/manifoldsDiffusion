ENV_NAME=diffusion

PYTHON_VERSION=3.8.5

.PHONY: all
all: install

.PHONY: dev_setup
dev_setup: create_env install_with_dev_tools

.PHONY: install
install: create_env install_dependencies

.PHONY: create_env
create_env:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y

.PHONY: install_dependencies
install_dependencies:
	conda run --name $(ENV_NAME) pip install .

.PHONY: install_with_dev_tools
install_with_dev_tools:
	conda run --name $(ENV_NAME) pip install --editable .[development,test]

.PHONY: clean
clean:
	conda env remove --name $(ENV_NAME) -y
