Get externally hosted data
DATASET_PATH="./datasets/squad/raw"
PROCESSED_DATASET_PATH = "./datasets/squad/processed"
mkdir -p $DATASET_PATH
mkdir -p $PROCESSED_DATASET_PATH
mkdir -p "./models"

# Get SQuAD train
echo "Get SQuAD train"
wget -O "$DATASET_PATH/SQuAD-v1.1-train.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
#Get SQuAD dev
echo "Get SQuAD train"
wget -O "$DATASET_PATH/SQuAD-v1.1-dev.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

echo "SQAUD dataset download completed"

# Download GloVe
GLOVE_DIR="embeddings/glove"
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR 

python -m spacy download en

echo "Processing squad dataset"
python make_dataset.py