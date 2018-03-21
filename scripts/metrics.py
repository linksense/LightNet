import numpy as np


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)

        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        acc_cls = tp / (sum_a1 + np.finfo(np.float32).eps)
        acc_cls = np.nanmean(acc_cls)

        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall_Acc': acc,
                'Mean_Acc': acc_cls,
                'FreqW_Acc': fwavacc,
                'Mean_IoU': mean_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


if __name__ == "__main__":
    n_class = 2
    score = RunningScore(n_class)

    label_true = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    label_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])

    score.update(label_true, label_pred)
    print(score.confusion_matrix)

