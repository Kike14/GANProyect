# GANProyect

## Description
This project aims to simulate stock price movements using a Wasserstein Generative Adversarial Network (WGAN) and evaluate a trading strategy through backtesting. The project compares the performance of an optimized trading strategy against a passive investment strategy, with different configurations of stop-loss and take-profit tested to maximize the **Calmar Ratio**.

## Technologies Used
- Python 3.12
- Libraries: NumPy, Pandas, Matplotlib, TensorFlow, TA-Lib, TQDM, Yahoo Finance (or others specified in `requirements.txt`)

## Prerequisites
1. Python 3.12 installed on your machine.

## Installation Instructions

## Steps for Windows:
1. Clone the repository:
   ```bash
   git clone https://github.com/Kike14/GANProyect

2. Create a virtual environment:
   bash
   python -m venv venv

3. Activate the virtual environment:
   bash
   .\venv\Scripts\activate
4. Upgrade pip:
   bash
   pip install --upgrade pip

5. Install dependencies:
   bash
   pip install -r requirements.txt

6. Run the main script:
   bash
   python main.py

## Steps for Mac:
1. Clone the repository:
   ```bash
   git clone https://github.com/Kike14/GANProyect

2. Create a virtual environment:
   bash
   python3 -m venv venv

3. Activate the virtual environment:
   bash
   source venv/bin/activate

4. Upgrade pip:
   bash
   pip install --upgrade pip

5. Install dependencies:
   bash
   pip install -r requirements.txt

6. Run the main script:
   bash
   python main.py 

## Project Structure:
GANProyect/
│
├── .idea/
├── .ipynb_checkpoints/
├── .venv/
├── fun/
│   ├── backtest.py
│   ├── data.py
│   └── WGAN.py
├── .gitignore
├── generador.keras
├── LICENSE
├── main.py
├── README.md
└── requirements.txt

## Contributing
Contributions are welcome. Please submit a pull request following the style guidelines and best practices.

## License
Copy this entire block and paste it into the `README.md` file in your PyCharm project. This version includes all necessary steps for both Windows and Mac, along with the full project description, structure, and license details.