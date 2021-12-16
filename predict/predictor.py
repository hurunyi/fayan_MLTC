import torch
import numpy as np
from utils.utils import model_device
from callback.progressbar import ProgressBar
from sklearn.metrics import f1_score
from train.metrics import MultiLabelReport


class Predictor(object):
    def __init__(self, model, logger, n_gpu, pair_model):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)
        self.pair_model = pair_model

    def predict(self, data, id2label):
        pbar = ProgressBar(n_total=len(data))
        all_logits = None
        y_true = None
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                if self.pair_model:
                    guid, label_ids, \
                    input_ids, input_mask, segment_ids, \
                    input_ids_2, input_mask_2, segment_ids_2 = batch
                    guid, logits = self.model(guid, input_ids, input_ids_2, input_mask, segment_ids,
                                              input_mask_2, segment_ids_2)
                else:
                    guid, input_ids, input_mask, segment_ids, label_ids = batch
                    guid, logits = self.model(guid, input_ids, input_mask, segment_ids)

                if y_true is None:
                    y_true = label_ids.detach().cpu().numpy()
                else:
                    y_true = np.concatenate(
                        [y_true, label_ids.detach().cpu().numpy()], axis=0)

                if logits.dim() == 1:
                    logits.unsqueeze(dim=0)
                guid = guid.detach().cpu().numpy()
                logits = logits.sigmoid().detach().cpu().numpy()
                if all_logits is None:
                    all_logits = logits
                    all_guid = guid
                else:
                    all_logits = np.concatenate([all_logits, logits], axis=0)
                    all_guid = np.concatenate([all_guid, guid], axis=0)
                pbar.batch_step(step=step, info={}, bar_type='Testing')

        if len(all_logits[0]) < len(y_true[0]):
            cut_len = len(y_true[0]) - len(all_logits[0])
            all_logits = np.append(all_logits, np.zeros((len(all_logits), cut_len)), axis=1)

        best_micro = 0
        best_macro = 0
        best_score = 0
        best_thresh = 0
        best_y_pred = None
        for threshold in [i * 0.01 for i in range(100)]:
            y_pred = (all_logits > threshold) * 1
            micro = f1_score(y_true, y_pred, average='micro')
            macro = f1_score(y_true, y_pred, average='macro')
            score = (micro + macro) / 2
            if score > best_score:
                best_micro = micro
                best_macro = macro
                best_score = score
                best_thresh = threshold
                best_y_pred = y_pred

        print("\nBest threshold: {}".format(best_thresh))
        print("\nScore: micro {}, macro {} Average {}".format(best_micro, best_macro, best_score))

        MultiLabelReport(id2label=id2label)(torch.from_numpy(all_logits), torch.from_numpy(y_true))

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_guid, all_logits, best_y_pred, y_true
