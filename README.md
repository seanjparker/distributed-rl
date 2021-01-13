# Distributed Reinforcement Learning

## Training

Running the following command launches the trainer in the selected framework using a list of worker sizes. By default it runs for 2, 4, 8, 16 and 32 workers.
```bash
$ ./launch -f [tf|torchr|ray] -e [number of epochs]
```