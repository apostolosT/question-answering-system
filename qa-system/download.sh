# # Get externally hosted data
# DATASET_PATH="./datasets/squad/raw"
# mkdir DATASET_PATH

# # Get SQuAD train
# echo "Get SQuAD train"
# wget -O "$DATASET_PATH/SQuAD-v1.1-train.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
# #Get SQuAD dev
# echo "Get SQuAD train"
# wget -O "$DATASET_PATH/SQuAD-v1.1-dev.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
# # python scripts/convert/squad.py "$DATASET_PATH/SQuAD-v1.1-train.json" "$DATASET_PATH/SQuAD-v1.1-train.txt"

# # Download official eval for SQuAD
# # curl "https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/" >  "./scripts/reader/official_eval.py"

# echo "SQAUD dataset download completed"

# # Download GloVe
# GLOVE_DIR="embeddings/glove"
# mkdir -p $GLOVE_DIR
# wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
# unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR 

# python -m spacy download en