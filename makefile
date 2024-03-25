.PHONY: style quality test

check_dirs := recommender tests api 

style: 
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix
quality:
	ruff format $(check_dirs) 
	ruff check $(check_dirs)
test: 
	pytest