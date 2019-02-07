# Adventures in ML
## How to run
### anaconda is used for the virtual environment.
1. Clone the repo.
2. Install anaconda if not already installed.
3. Setup the environment using the following commands:
```bash
conda env create -f environment.yml
```
4. Verify that the new environment was installed correctly:
```bash
conda list
```
5. Activate the environment (change 'myenv' to the name of the environment):
..* On Windows, in your Anaconda Prompt run:
```bash
activate myenv
```
..* On macOS and Linux
```bash
source activate myenv
```
6. Navigate into desired folder and open the jupyter notebook using command:
..* 
```bash
jupyter notebook
```
7. To deactivate the environment
..* On Windows, in your Anaconda Prompt run:
```bash
deactivate
```
..* On macOS and Linux
```bash
source deactivate
```
