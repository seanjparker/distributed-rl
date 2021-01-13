# Distributed Reinforcement Learning

## Requirements

- Python 3 (>=3.7)
- Pipenv

## Setup
As long as you have Python3 and Pipenv installed, run the following command: 

```shell script
$ pipenv install
```

Pipenv will automatically create and install all the required dependencies for the project.

## Training

Running the following command launches the trainer in the selected framework using a list of worker sizes. By default it runs for 2, 4, 8, 16 and 32 workers.
```shell script
$ pipenv shell # Activate the virtualenv created by Pipenv
$ ./launch -f [tf|torch|ray] -e [number of epochs]
```