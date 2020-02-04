import argparse
import logging
import traceback
from udify.dataset_readers.conll18_ud_eval import evaluate, load_conllu_file, UDError
from udify.util import save_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--gold", type=str, help="")
parser.add_argument("--pred", type=str, help="")
parser.add_argument("--output", type=str, help="")
args = parser.parse_args()


logger = logging.getLogger(__name__)


try:
    evaluation = evaluate(load_conllu_file(args.gold), load_conllu_file(args.pred))
    save_metrics(evaluation, args.output)
except UDError:
    logger.warning(f"Failed to evaluate {args.pred}")
    traceback.print_exc()
