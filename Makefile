### ---------------   Lint  --------------- ###

pylint:
	pylint --rcfile=pylint.conf main.py components/*.py

lint:
	make pylint

### ---------------   Format  --------------- ###

format:
	black main.py components/*.py


### ---------------   Run  --------------- ###

run:
	python3 -m streamlit run main.py

run-dev:
	python3 -m streamlit run main.py -- --dev # if you have your OpenAI API key in the .env file