## Currency Detection System
### Overview
- YOLO-based Detection: Detects all banknotes in an image and localizes them with bounding boxes.
- CNN-based Classification: Each detected note is classified by country and denomination using an enhanced attention-based multi-task CNN.

### Goal
- Detect and classify banknotes from multiple countries.
- Predict both country and denomination accurately, even when multiple notes are present in a single image.

### Approach
- Shared CNN backbone extracts features.
- Two heads predict country and denomination separately.

### why multi-headed-cnn?
- Using one head would force the network to predict country and denomination together as a single class. This leads to too many classes, reduces task-specific focus, and makes the model less flexible and accurate. 
- A multi-headed CNN allows one shared backbone while letting separate heads specialize for country and denomination, improving efficiency, accuracy, and scalability.

### project structure
```bash
currency-detection/
├─ architecture/           # CNN  model definitions
├─ currency_detector/      # fast api app for localizationand clssification
├─ data/                   # Training and testing datasets
├─ localization/           # Object detection notebook and results
├─ models/                 # Trained model weights and checkpoints
├─ .gitignore              # Git ignore file and evaluation
├─ data preparation.ipynb  # Notebook for dataset preprocessing
├─ notebookbafda3ade1.ipynb # Experimental notebook
└─ README.md               # Project documentation
```
### usuage
``` bash
git clone https://github.com/your-username/currency-detection.git
cd currency-detection
```
- Install dependencies
```bash
pip install -r requirements.txt
```
- Run the FastAPI server

Inside the currency_detector/ directory:
```bash
cd currency_detector
uvicorn main:app --reload
```
4. Access the API

Once the server is running, open in your browser:
```bash
http://127.0.0.1:8000
```