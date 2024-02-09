__author__ = 'djgagne'

import unittest
from hagelslag.evaluation.ProbabilityMetrics import DistributedReliability, DistributedROC, DistributedCRPS
import numpy as np


class TestProbabilityMetrics(unittest.TestCase):
    def setUp(self):
        self.num_forecasts = 1000
        self.forecasts = dict(perfect=np.concatenate((np.ones(self.num_forecasts// 2), np.zeros(self.num_forecasts// 2))),
                              random=np.random.random(self.num_forecasts))
        self.observations= dict(perfect=self.forecasts['perfect'],
                                random=self.forecasts['perfect'])
        self.thresholds = np.arange(0, 1.2, 0.1)
        self.obs_threshold = 0.5
        return

    def test_reliability(self):
        perfect_rel = DistributedReliability(self.thresholds, self.obs_threshold)
        perfect_rel.update(self.forecasts["perfect"], self.observations["perfect"])
        random_rel = DistributedReliability(self.thresholds, self.obs_threshold)
        random_rel.update(self.forecasts["random"], self.observations["random"])
        perfect_components = perfect_rel.brier_score_components()
        self.assertEqual(perfect_rel.frequencies["Total_Freq"].sum(), self.num_forecasts,
                         msg="Total Frequency does not match number of forecasts.")
        self.assertEqual(perfect_rel.frequencies["Positive_Freq"].sum(), self.num_forecasts / 2,
                         msg="Positive Frequency does not match number of positive forecasts.")
        self.assertEqual(perfect_components[1], perfect_components[2], "Resolution does not equal uncertainty.")
        self.assertEqual(perfect_rel.brier_score(), 0,
                         msg="Perfect Brier score is {0:0.3f}".format(perfect_rel.brier_score()))
        self.assertGreater(random_rel.brier_score(), perfect_rel.brier_score(),
                           msg="Perfect (BS={0:0.3f}) has worse score than random (BS={1:0.3f})".format(
                               perfect_rel.brier_score(), random_rel.brier_score()))
        perfect_rel_copy = DistributedReliability(input_str=str(perfect_rel))
        self.assertEqual(perfect_rel.brier_score(), perfect_rel_copy.brier_score(),
                         msg="Brier Score of copy {0} does not match original {1}".format(perfect_rel.brier_score(),
                                                                                          perfect_rel_copy.brier_score()
                                                                                          ))
        pbss = perfect_rel.brier_skill_score()
        cpbss = perfect_rel_copy.brier_skill_score()
        self.assertEqual(pbss, cpbss,
                         msg="BSS of copy {0} does not match original {1}".format(pbss, cpbss))
        self.assertLessEqual(perfect_rel.frequencies["Positive_Freq"].sum(),
                             perfect_rel.frequencies["Total_Freq"].sum(),
                             msg="There are more perfect positives than total events")
        self.assertLessEqual(random_rel.frequencies["Positive_Freq"].sum(),
                             random_rel.frequencies["Total_Freq"].sum(),
                             msg="There are more random positives than total events")
        perfect_sum = perfect_rel + perfect_rel
        mixed_sum = perfect_rel + random_rel
        self.assertEqual(perfect_rel.brier_score(), perfect_sum.brier_score(),
                         msg="Summed perfect brier score not equal to perfect brier score")
        self.assertLess(perfect_sum.brier_score(), mixed_sum.brier_score(),
                           msg="Perfect brier score greater than mixed brier score")

    def test_roc(self):
        perfect_roc = DistributedROC(self.thresholds, self.obs_threshold)
        perfect_roc.update(self.forecasts["perfect"], self.observations["perfect"])
        perfect_auc = perfect_roc.auc()
        random_roc = DistributedROC(self.thresholds, self.obs_threshold)
        random_roc.update(self.forecasts["random"], self.observations["random"])
        random_auc = random_roc.auc()
        self.assertEqual(perfect_auc, 1, msg="Perfect AUC not 1, is actually {0:0.2f}".format(perfect_auc))
        self.assertLessEqual(np.abs(random_auc - 0.5), 0.1,
                             msg="Random AUC not 0.5, actually {0:0.3f}".format(random_auc))
        self.assertGreater(perfect_auc, random_auc, msg="Perfect AUC is not greater than random.")

    def test_crps(self):
        thresholds = np.arange(100)
        obs = np.zeros((1000, 100))
        for o in range(obs.shape[1]):
            ob_ix = np.reshape(np.arange(0, 1000, 100) + o, (10, 1))
            obs[ob_ix, thresholds[o:].reshape(1, 100 - o)] = 1
        perfect_crps = DistributedCRPS(thresholds=thresholds)
        perfect_crps.update(obs, obs)
        self.assertEqual(perfect_crps.crps(), 0, "CRPS for perfect forecast is not 0")
        self.assertGreater(perfect_crps.crps_climo(), 0, "Climo CRPS is greater than 0")
        self.assertLess(perfect_crps.crps_climo(), 1, "Climo CRPS is less than 1")
        self.assertEqual(perfect_crps.crpss(), 1,
                         "CRPSS for perfect forecast is not 1, is {0}".format(perfect_crps.crpss()))
        crps_copy = DistributedCRPS(input_str=str(perfect_crps))
        self.assertEqual(crps_copy.crps(), 0, "CRPS copy is not 0")
        self.assertEqual(crps_copy.crpss(), 1, "CRPSS copy is not 1")
        crps_sum = perfect_crps + perfect_crps
        self.assertEqual(crps_sum.crps(), 0, "CRPS sum is not 0")
        self.assertEqual(crps_sum.crpss(), 1, "CRPSS sum is not 1")
