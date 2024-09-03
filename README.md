# WeatherTag mini coding project

A small webapp and a tool that annotates [Wikipedia Current Events](https://en.wikipedia.org/wiki/Portal:Current_events) with weather information from [Open Weather Map data](https://openweathermap.org/bulk).

## Setup

### 1. Install prerequisites

Make sure your system has a working `conda` command:

```bash
brew install miniforge  # miniconda alternative that kinda works better on arm64 macOS (M1 Macs and beyond)
```

Then you can install a couple of requirements for Python:

```bash
./environment.yml

# or:
conda env update -f ./environment.yml
```

### 2. Download

Download `daily_14.json.gz` from <http://bulk.openweathermap.org/sample/> and put it in the `data`folder.

### 3. Setup AI capabilities

If you have an OpenAI API key and to unlock full capabilities of info extraction, you can create a `.env` file and insert the line:

```.env
OPENAI_API_KEY=<your-api-key>
```

### 4. Launch the streamlit app.

```bash
./weathertag.py

# or:
conda run -n weathertag_env  streamlit run weathertag.py
```

That should open a web page in your browser.

## Useful Commands

### Lint

Run `make lint` to launch the lints for both the repository.

### Format

Run `make format` to launch the `black`formatter on the project.

### Run stream

Run `make run` to launch the streamlit app.

## Useful Links

- [Commit convention (with emojis)](https://github.com/kazupon/git-commit-message-convention/blob/master/README.md)
