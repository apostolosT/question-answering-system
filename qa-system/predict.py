import torch
import pickle
from pytorch_lightning import Trainer, seed_everything
import logging
import warnings
warnings.filterwarnings('ignore')
from src.models.document_reader_qa.dataset import PredictDataset
from src.models.document_reader_qa.model import QuAModel
from src.models.document_reader_qa.model_lit import QuALit
from src.models.document_reader_qa.train import evaluate
from src.data.vectorize_single_example import vectorize_example

seed_everything(42, workers=True)

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

MODEL_CPT_PAT = "models/dr_model/version_1/checkpoints/latest-epoch=4-step=27000.ckpt"

checkpoint = torch.load(MODEL_CPT_PAT)

with open("datasets/squad/processed/idx2word.pickle", 'rb') as file:
    idx2word = pickle.load(file)
# Model
HIDDEN_DIM = 128
EMB_DIM = 300
NUM_LAYERS = 3
NUM_DIRECTIONS = 2
DROPOUT = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = QuAModel(
    HIDDEN_DIM,
    EMB_DIM,
    NUM_LAYERS,
    NUM_DIRECTIONS,
    DROPOUT,
    device,
    "datasets/squad/processed/weights-matrix.npy")


model_lit = QuALit(
    model,
    idx2word=idx2word,
    evaluate_func=evaluate
)
model_lit.model.load_state_dict(checkpoint, strict=False)

g = torch.Generator()

context = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."
question = "When did Victoria enact its constitution?"
raw_input = {
    "raw_context": context,
    "raw_question": question
}
test_input = vectorize_example(**raw_input)
test_set = PredictDataset(test_input)

predict_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    generator=g,
)

trainer = Trainer(deterministic=True)
predictions = trainer.predict(model_lit, dataloaders=predict_dataloader)
print(predictions[0])
