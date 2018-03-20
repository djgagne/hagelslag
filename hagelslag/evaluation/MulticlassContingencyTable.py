import numpy as np
__author__ = 'David John Gagne <djgagne@ou.edu>'


def main():
    # Contingency Table from Wilks (2011) Table 8.3
    table = np.array([[50, 91, 71],
                      [47, 2364, 170],
                      [54, 205, 3288]])
    mct = MulticlassContingencyTable(table, n_classes=table.shape[0],
                                     class_names=np.arange(table.shape[0]).astype(str))
    print(mct.peirce_skill_score())
    print(mct.gerrity_score())


class MulticlassContingencyTable(object):
    """
    This class is a container for a contingency table containing more than 2 classes.
    The contingency table is stored in table as a numpy array with the rows corresponding to forecast categories,
    and the columns corresponding to observation categories.
    """
    def __init__(self, table=None, n_classes=2, class_names=("1", "0")):
        self.table = table
        self.n_classes = n_classes
        self.class_names = class_names
        if table is None:
            self.table = np.zeros((self.n_classes, self.n_classes), dtype=int)

    def __add__(self, other):
        assert self.n_classes == other.n_classes, "Number of classes does not match"
        return MulticlassContingencyTable(self.table + other.table,
                                          n_classes=self.n_classes,
                                          class_names=self.class_names)

    def peirce_skill_score(self):
        """
        Multiclass Peirce Skill Score (also Hanssen and Kuipers score, True Skill Score)
        """
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (no * no).sum() / n ** 2)

    def gerrity_score(self):
        """
        Gerrity Score, which weights each cell in the contingency table by its observed relative frequency.
        :return:
        """
        k = self.table.shape[0]
        n = float(self.table.sum())
        p_o = self.table.sum(axis=0) / n
        p_sum = np.cumsum(p_o)[:-1]
        a = (1.0 - p_sum) / p_sum
        s = np.zeros(self.table.shape, dtype=float)
        for (i, j) in np.ndindex(*s.shape):
            if i == j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:j]) + np.sum(a[j:k-1]))
            elif i < j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:i]) - (j - i) + np.sum(a[j:k-1]))
            else:
                s[i, j] = s[j, i]
        return np.sum(self.table / float(self.table.sum()) * s)

    def heidke_skill_score(self):
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (nf * no).sum() / n ** 2)


if __name__ == "__main__":
    main()
