#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:50:59 2021

@author: modal
"""

# %%

from fastai.tabular.all import *
# from torch.nn import functional as F
from .metrics import within_group_variance, between_group_variance

from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# %%
# Metric handler in case the target contains also the group


def metric_group_handler(pred, targ, m, do_group=False):
    if do_group:
        return m(pred, targ[:, 1:])
    return m(pred, targ)


# %%
# Loss functions


def r2_loss(output, targ):
    targ_mean = torch.mean(targ)
    ss_tot = torch.sum((targ - targ_mean) ** 2)
    ss_res = torch.sum((targ - output) ** 2)
    r2 = 1 - (ss_res + 1e-6) / (ss_tot + 1e-6)
    return 1 - r2


class R2Loss:
    def __init__(self):
        pass

    def __call__(self, pred, targ):
        return r2_loss(pred, targ)


def sigma_loss(output, targ):
    resid = targ - output
    resid_mean = torch.mean(resid)
    return torch.mean((resid - resid_mean) ** 2)


class SigmaLoss:
    def __init__(self):
        pass

    def __call__(self, pred, targ):
        return sigma_loss(pred, targ)


class MSESigmaTotalLoss:
    def __init__(self, coeff_mse=1, coeff_sigma=1):
        store_attr()
        self.mseloss = nn.MSELoss()
        self.sigmaloss = SigmaLoss()

    def __call__(self, pred, targ):
        return self.coeff_mse * self.mseloss(pred, targ) \
            + self.coeff_sigma * self.sigmaloss(pred, targ)


class MSESigmaLoss:
    def __init__(
            self, coeff_mse=1, coeff_sigma_between=1, coeff_sigma_within=1):
        store_attr()

    def __call__(self, pred, targ):
        y_group = targ[:, 0].int()
        y_true = targ[:, 1:]

        y_pred = pred
        y_resid = y_true - y_pred

        return self.coeff_mse * F.mse_loss(y_true, y_pred) \
            + self.coeff_sigma_between * between_group_variance(
                y_resid, y_group) \
            + self.coeff_sigma_within * within_group_variance(
                y_resid, y_group)


# %%


class LinBnDrop2(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(
            self, n_in, n_out, bn=True, p=0., act=None, lin_first=False,
            bias=True):
        layers = [
            BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn or bias)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


class TabularModelGMPE(Module):
    "Basic model for tabular data."
    def __init__(
            self, n_cont, out_sz, layers, ps=None, y_range=None, use_bn=True,
            bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
            lin_first=True, biases=None):
        ps = ifnone(ps, [0] * len(layers))
        if not is_listy(ps):
            ps = [ps] * len(layers)
        biases = ifnone(biases, [True] * len(layers))
        if not is_listy(biases):
            biases = [biases] * len(layers)
        # self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        # self.emb_drop = nn.Dropout(embed_p)
        # self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        # n_emb = sum(e.embedding_dim for e in self.embeds)
        # self.n_emb,self.n_cont = n_emb,n_cont
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        self.n_cont = n_cont
        sizes = [n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes) - 2)] + [None]
        _layers = [
            LinBnDrop2(
                sizes[i], sizes[i + 1],
                bn=use_bn and (i != len(actns) - 1 or bn_final),
                p=p, act=a, lin_first=lin_first)
            for i, (p, a) in enumerate(zip(ps + [0.], actns))]
        if y_range is not None:
            _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.bn_cont is not None:
            x_cont = self.bn_cont(x_cont)
        return self.layers(x_cont)


@delegates(Learner.__init__)
def tabular_learner_GMPE(
        dls, layers=None, config=None, n_out=None, y_range=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` "
    "created using the remaining params."
    if config is None:
        config = tabular_config()
    if layers is None:
        layers = [200, 100]
    if n_out is None:
        n_out = 1
    assert n_out, "`n_out` is not defined, and could not be inferred from " \
        "data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config:
        y_range = config.pop('y_range')
    model = TabularModelGMPE(
        len(dls.cont_names), n_out, layers, y_range=y_range, **config)
    return TabularLearner(dls, model, **kwargs)

# %%


class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and "
    "loads it at the end."
    _only_train_loop, order = True, TrackerCallback.order + 1

    def __init__(
            self, monitor='valid_loss', comp=None, min_delta=0.,
            fname='model', every_epoch=False, at_end=False,
            with_opt=False, reset_on_fit=True):
        super().__init__(
            monitor=monitor, comp=comp, min_delta=min_delta,
            reset_on_fit=reset_on_fit)
        assert not (every_epoch and at_end), \
            "every_epoch and at_end cannot both be set to True"
        # keep track of file path for loggers
        self.last_saved_path = None
        store_attr('fname,every_epoch,at_end,with_opt')

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            if (self.epoch % self.every_epoch) == 0:
                self._save(f'{self.fname}_{self.epoch}')
        else:  # every improvement
            super().after_epoch()
            if self.new_best:
                print(f'\rBetter model found at epoch {self.epoch} with '
                      f'{self.monitor} value: {self.best:.4f}.', end='')
                self._save(f'{self.fname}')

    def after_fit(self, **kwargs):
        "Load the best model."
        if self.at_end:
            self._save(f'{self.fname}')
        elif not self.every_epoch:
            self.learn.load(f'{self.fname}', with_opt=self.with_opt)

# %%


def build_coefficients(
        dff: pd.DataFrame, s_target: Optional[str] = None,
        ohe: bool = True):
    if s_target is None:
        return dff

    df_s = dff[[s_target]].copy()
    if ohe:
        s_target_ohes = [f'{s_target}_{j}' for j in range(2)]
        df_s[s_target_ohes[0]] = (df_s[s_target] == -1).astype('float')
        df_s[s_target_ohes[1]] = (df_s[s_target] == 1).astype('float')
        s_target_cols = s_target_ohes
    else:
        df_s[s_target] = df_s['s'].astype('float')
        s_target_cols = [s_target]
    return df_s[s_target_cols]


def is_fitted(estimator):
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        return False
    return True


# Temporary class to hold everything needed for the prediction.
# TODO move everything to fastai
class LearnerAll:
    def __init__(
            self, learner: Learner, scalerX: TransformerMixin,
            scalerY: TransformerMixin, features_cols: list[str],
            s_target: Optional[str], target_col: str, group_col: str,
            group_loss: bool, ohe: bool) -> None:
        store_attr()

    def build_s(self, dff):
        df_transf = dff.copy()
        features_cols_ = self.features_cols
        s_target = self.s_target
        if s_target is None:
            return df_transf

        features_cols_nos = [col for col in features_cols_ if col != s_target]
        df_s = build_coefficients(df_transf, s_target, ohe=self.ohe)

        features_cols_ = features_cols_nos + list(df_s.columns)
        df_transf = pd.concat([df_transf, df_s], axis=1)
        return df_transf, features_cols_

    def predict(
            self, dff: pd.DataFrame, target_pred_col: Optional[str] = None):
        target_col_ = self.target_col
        group_col_ = self.group_col
        scalerX_ = self.scalerX
        scalerY_ = self.scalerY
        learner_ = self.learner
        target_pred_col_ = target_pred_col
        if target_pred_col is None:
            target_pred_col_ = f'{target_col_}_pred'

        assert is_fitted(scalerX_) and is_fitted(scalerY_)

        df_transf, features_cols_ = self.build_s(dff)
        df_transf[features_cols_] = scalerX_.transform(
            df_transf[features_cols_])
        if self.group_loss:
            df_transf[group_col_] = df_transf[group_col_].astype('float')

        dl_dff = learner_.dls.test_dl(df_transf)

        dff_target = df_transf[[features_cols_[0]]].copy()
        with ContextManagers(
                [learner_.no_logging(), learner_.no_bar()]):
            dff_target[target_pred_col_] = learner_.get_preds(
                dl=dl_dff)[0].numpy()

        dff_target[[target_pred_col_]] = \
            scalerY_.inverse_transform(dff_target[[target_pred_col_]])
        return dff_target[target_pred_col_]
