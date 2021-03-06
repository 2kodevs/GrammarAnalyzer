.DEFAULT_GOAL 	:= run

run: ## Run the proyect
	python main.py

install: ## Install dependencies
	sudo apt-get install graphviz
	pip install pydot
	pip install eel

view: ## display the Makefile
	@cat Makefile

edit: ## open the Makefile with `code`
	@code Makefile

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

