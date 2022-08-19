import torch
import pickle
from pytorch_lightning import Trainer, seed_everything
import logging
import warnings

from src import paths
from src.data.vectorize_single_example import vectorize_example
from src.data.utils import normalize_text, get_context_tokens, normalize_spaces
from src.train import evaluate
from src.models.qa_model.model_lit import QuALit
from src.models.qa_model.model import QuAModel
from src.models.qa_model.dataset import PredictDataset
from configs.model import params, device

warnings.filterwarnings('ignore')


random_seed = 1  # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(random_seed)

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

with open(paths.IDX2WORD_PATH, 'rb') as file:
    idx2word = pickle.load(file)


# Init Pytorch Model
model = QuAModel(
    params
)
# Init Lightning Model
with torch.no_grad():
    model_lit = QuALit(
        model,
        idx2word=idx2word,
        evaluate_func=evaluate
    )
# Load Checkpoint
checkpoint = torch.load(paths.MODEL_CKPT_PATH)
model_lit.model.load_state_dict(checkpoint, strict=False)

g = torch.Generator()
g.manual_seed(1)

def predict_answer(context, question):
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

    trainer = Trainer(deterministic=True, enable_progress_bar=False)
    pred: list = trainer.predict(model_lit, dataloaders=predict_dataloader)
    pred = pred[0]

    return pred

context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24-10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California"
question = "What is the AFC short for?"

# ans = predict_answer(context, question)
# print(ans)
