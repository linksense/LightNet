import math
import numpy as np
import torch
import copy
import logging
import os
import pickle as cp

# eps for numerical stability
eps = 1e-6


class YFOptimizer(object):
    def __init__(self, var_list, lr=0.001, mu=0.0, clip_thresh=None, weight_decay=0.0,
                 beta=0.999, curv_win_width=20, zero_debias=True, sparsity_debias=False, delta_mu=0.0,
                 auto_clip_fac=None, force_non_inc_step=False, h_max_log_smooth=True, h_min_log_smooth=True,
                 checkpoint_interval=1000, verbose=False, adapt_clip=True, stat_protect_fac=100.0,
                 catastrophic_move_thresh=100.0,
                 use_disk_checkpoint=False, checkpoint_dir='./YF_workspace'):
        '''
        clip thresh is the threshold value on ||lr * gradient||
        delta_mu can be place holder/variable/python scalar. They are used for additional
        momentum in situations such as asynchronous-parallel training. The default is 0.0
        for basic usage of the optimizer.
        Args:
          lr: python scalar. The initial value of learning rate, we use 1.0 in our paper.
          mu: python scalar. The initial value of momentum, we use 0.0 in our paper.
          clip_thresh: python scalar. The manaully-set clipping threshold for tf.clip_by_global_norm.
            if None, the automatic clipping can be carried out. The automatic clipping
            feature is parameterized by argument auto_clip_fac. The auto clip feature
            can be switched off with auto_clip_fac = None
          beta: python scalar. The smoothing parameter for estimations.
          sparsity_debias: gradient norm and curvature are biased to larger values when
          calculated with sparse gradient. This is useful when the model is very sparse,
          e.g. LSTM with word embedding. For non-sparse CNN, turning it off could slightly
          accelerate the speed.
          delta_mu: for extensions. Not necessary in the basic use.
          force_non_inc_step: in some very rare cases, it is necessary to force ||lr * gradient||
          to be not increasing dramatically for stableness after some iterations.
          In practice, if turned on, we enforce lr * sqrt(smoothed ||grad||^2)
          to be less than 2x of the minimal value of historical value on smoothed || lr * grad ||.
          This feature is turned off by default.
          checkpoint_interval: interval to do checkpointing. For potential recovery from crashing.
          stat_protect_fac: a loose hard adaptive threshold over ||grad||^2. It is to protect stat
          from being destropied by exploding gradient.
        Other features:
          If you want to manually control the learning rates, self.lr_factor is
          an interface to the outside, it is an multiplier for the internal learning rate
          in YellowFin. It is helpful when you want to do additional hand tuning
          or some decaying scheme to the tuned learning rate in YellowFin.
          Example on using lr_factor can be found here:
          https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/pytorch-cifar/main.py#L109
        '''
        self._lr = lr
        self._mu = mu
        self._lr_t = lr
        self._mu_t = mu
        # we convert var_list from generator to list so that
        # it can be used for multiple times
        self._var_list = list(var_list)
        self._clip_thresh = clip_thresh
        self._auto_clip_fac = auto_clip_fac
        self._beta = beta
        self._curv_win_width = curv_win_width
        self._zero_debias = zero_debias
        self._sparsity_debias = sparsity_debias
        self._force_non_inc_step = force_non_inc_step
        self._optimizer = torch.optim.SGD(self._var_list, lr=self._lr,
                                          momentum=self._mu, weight_decay=weight_decay)
        self._iter = 0
        # global states are the statistics
        self._global_state = {}

        # for decaying learning rate and etc.
        self._lr_factor = 1.0

        # smoothing options
        self._h_max_log_smooth = h_max_log_smooth
        self._h_min_log_smooth = h_min_log_smooth

        # checkpoint interval
        self._checkpoint_interval = checkpoint_interval

        self._verbose = verbose
        if self._verbose:
            logging.debug('Verbose mode with debugging info logged.')

        # clip exploding gradient
        self._adapt_clip = adapt_clip
        self._exploding_grad_clip_thresh = 1e3
        self._exploding_grad_clip_target_value = 1e3
        self._stat_protect_fac = stat_protect_fac
        self._catastrophic_move_thresh = catastrophic_move_thresh
        self._exploding_grad_detected = False

        # workspace creation
        self._use_disk_checkpoint = use_disk_checkpoint
        self._checkpoint_dir = checkpoint_dir
        if use_disk_checkpoint:
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
            self._checkpoint_file = "checkpoint_pid_" + str(os.getpid())

    def state_dict(self):
        # for checkpoint saving
        sgd_state_dict = self._optimizer.state_dict()
        # for recover model internally in case of numerical issue
        model_state_list = [p.data \
                            for group in self._optimizer.param_groups for p in group['params']]
        global_state = self._global_state
        lr_factor = self._lr_factor
        iter = self._iter
        lr = self._lr
        mu = self._mu
        clip_thresh = self._clip_thresh
        beta = self._beta
        curv_win_width = self._curv_win_width
        zero_debias = self._zero_debias
        h_min = self._h_min
        h_max = self._h_max

        return {
            "sgd_state_dict": sgd_state_dict,
            "model_state_list": model_state_list,
            "global_state": global_state,
            "lr_factor": lr_factor,
            "iter": iter,
            "lr": lr,
            "mu": mu,
            "clip_thresh": clip_thresh,
            "beta": beta,
            "curv_win_width": curv_win_width,
            "zero_debias": zero_debias,
            "h_min": h_min,
            "h_max": h_max
        }

    def load_state_dict(self, state_dict):
        # for checkpoint saving
        self._optimizer.load_state_dict(state_dict['sgd_state_dict'])
        # for recover model internally if any numerical issue happens
        param_id = 0
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                p.data.copy_(state_dict["model_state_list"][param_id])
                param_id += 1
        self._global_state = state_dict['global_state']
        self._lr_factor = state_dict['lr_factor']
        self._iter = state_dict['iter']
        self._lr = state_dict['lr']
        self._mu = state_dict['mu']
        self._clip_thresh = state_dict['clip_thresh']
        self._beta = state_dict['beta']
        self._curv_win_width = state_dict['curv_win_width']
        self._zero_debias = state_dict['zero_debias']
        self._h_min = state_dict["h_min"]
        self._h_max = state_dict["h_max"]
        return

    def load_state_dict_perturb(self, state_dict):
        # for checkpoint saving
        self._optimizer.load_state_dict(state_dict['sgd_state_dict'])
        # for recover model internally if any numerical issue happens
        param_id = 0
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                p.data.copy_(state_dict["model_state_list"][param_id])
                p.data += 1e-8
                param_id += 1
        self._global_state = state_dict['global_state']
        self._lr_factor = state_dict['lr_factor']
        self._iter = state_dict['iter']
        self._lr = state_dict['lr']
        self._mu = state_dict['mu']
        self._clip_thresh = state_dict['clip_thresh']
        self._beta = state_dict['beta']
        self._curv_win_width = state_dict['curv_win_width']
        self._zero_debias = state_dict['zero_debias']
        self._h_min = state_dict["h_min"]
        self._h_max = state_dict["h_max"]
        return

    def set_lr_factor(self, factor):
        self._lr_factor = factor
        return

    def get_lr_factor(self):
        return self._lr_factor

    def zero_grad(self):
        self._optimizer.zero_grad()
        return

    def zero_debias_factor(self):
        return 1.0 - self._beta ** (self._iter + 1)

    def zero_debias_factor_delay(self, delay):
        # for exponentially averaged stat which starts at non-zero iter
        return 1.0 - self._beta ** (self._iter - delay + 1)

    def curvature_range(self):
        global_state = self._global_state
        if self._iter == 0:
            global_state["curv_win"] = torch.FloatTensor(self._curv_win_width, 1).zero_()
        curv_win = global_state["curv_win"]
        grad_norm_squared = self._global_state["grad_norm_squared"]
        # curv_win[self._iter % self._curv_win_width] = np.log(grad_norm_squared + eps)
        curv_win[self._iter % self._curv_win_width] = grad_norm_squared
        valid_end = min(self._curv_win_width, self._iter + 1)
        # we use running average over log scale, accelerating
        # h_max / min in the begining to follow the varying trend of curvature.
        beta = self._beta
        if self._iter == 0:
            global_state["h_min_avg"] = 0.0
            global_state["h_max_avg"] = 0.0
            self._h_min = 0.0
            self._h_max = 0.0
        if self._h_min_log_smooth:
            global_state["h_min_avg"] = \
                global_state["h_min_avg"] * beta + (1 - beta) * torch.min(np.log(curv_win[:valid_end] + eps))
        else:
            global_state["h_min_avg"] = \
                global_state["h_min_avg"] * beta + (1 - beta) * torch.min(curv_win[:valid_end])
        if self._h_max_log_smooth:
            global_state["h_max_avg"] = \
                global_state["h_max_avg"] * beta + (1 - beta) * torch.max(np.log(curv_win[:valid_end] + eps))
        else:
            global_state["h_max_avg"] = \
                global_state["h_max_avg"] * beta + (1 - beta) * torch.max(curv_win[:valid_end])
        if self._zero_debias:
            debias_factor = self.zero_debias_factor()
            if self._h_min_log_smooth:
                self._h_min = np.exp(global_state["h_min_avg"] / debias_factor)
            else:
                self._h_min = global_state["h_min_avg"] / debias_factor
            if self._h_max_log_smooth:
                self._h_max = np.exp(global_state["h_max_avg"] / debias_factor)
            else:
                self._h_max = global_state["h_max_avg"] / debias_factor
        else:
            if self._h_min_log_smooth:
                self._h_min = np.exp(global_state["h_min_avg"])
            else:
                self._h_min = global_state["h_min_avg"]
            if self._h_max_log_smooth:
                self._h_max = np.exp(global_state["h_max_avg"])
            else:
                self._h_max = global_state["h_max_avg"]
        if self._sparsity_debias:
            self._h_min *= self._sparsity_avg
            self._h_max *= self._sparsity_avg
        return

    def grad_variance(self):
        global_state = self._global_state
        beta = self._beta
        self._grad_var = np.array(0.0, dtype=np.float32)
        for group_id, group in enumerate(self._optimizer.param_groups):
            for p_id, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self._optimizer.state[p]
                if self._iter == 0:
                    state["grad_avg"] = grad.new().resize_as_(grad).zero_()
                    state["grad_avg_squared"] = 0.0
                state["grad_avg"].mul_(beta).add_(1 - beta, grad)
                self._grad_var += torch.sum(state["grad_avg"] * state["grad_avg"])

        if self._zero_debias:
            debias_factor = self.zero_debias_factor()
        else:
            debias_factor = 1.0

        self._grad_var /= -(debias_factor ** 2)
        self._grad_var += global_state['grad_norm_squared_avg'] / debias_factor
        # in case of negative variance: the two term are using different debias factors
        self._grad_var = max(self._grad_var, eps)
        if self._sparsity_debias:
            self._grad_var *= self._sparsity_avg
        return

    def dist_to_opt(self):
        global_state = self._global_state
        beta = self._beta
        if self._iter == 0:
            global_state["grad_norm_avg"] = 0.0
            global_state["dist_to_opt_avg"] = 0.0
        global_state["grad_norm_avg"] = \
            global_state["grad_norm_avg"] * beta + (1 - beta) * math.sqrt(global_state["grad_norm_squared"])
        global_state["dist_to_opt_avg"] = \
            global_state["dist_to_opt_avg"] * beta \
            + (1 - beta) * global_state["grad_norm_avg"] / (global_state['grad_norm_squared_avg'] + eps)
        if self._zero_debias:
            debias_factor = self.zero_debias_factor()
            self._dist_to_opt = global_state["dist_to_opt_avg"] / debias_factor
        else:
            self._dist_to_opt = global_state["dist_to_opt_avg"]
        if self._sparsity_debias:
            self._dist_to_opt /= (np.sqrt(self._sparsity_avg) + eps)
        return

    def grad_sparsity(self):
        global_state = self._global_state
        if self._iter == 0:
            global_state["sparsity_avg"] = 0.0
        non_zero_cnt = 0.0
        all_entry_cnt = 0.0
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_non_zero = grad.nonzero()
                if grad_non_zero.dim() > 0:
                    non_zero_cnt += grad_non_zero.size()[0]
                all_entry_cnt += torch.numel(grad)
        beta = self._beta
        global_state["sparsity_avg"] = beta * global_state["sparsity_avg"] \
                                       + (1 - beta) * non_zero_cnt / float(all_entry_cnt)
        self._sparsity_avg = \
            global_state["sparsity_avg"] / self.zero_debias_factor()

        if self._verbose:
            logging.debug("sparsity %f, sparsity avg %f", non_zero_cnt / float(all_entry_cnt), self._sparsity_avg)

        return

    def lr_grad_norm_avg(self):
        # this is for enforcing lr * grad_norm not
        # increasing dramatically in case of instability.
        #  Not necessary for basic use.
        global_state = self._global_state
        beta = self._beta
        if "lr_grad_norm_avg" not in global_state:
            global_state['grad_norm_squared_avg_log'] = 0.0
        global_state['grad_norm_squared_avg_log'] = \
            global_state['grad_norm_squared_avg_log'] * beta \
            + (1 - beta) * np.log(global_state['grad_norm_squared'] + eps)
        if "lr_grad_norm_avg" not in global_state:
            global_state["lr_grad_norm_avg"] = \
                0.0 * beta + (1 - beta) * np.log(self._lr * np.sqrt(global_state['grad_norm_squared']) + eps)
            # we monitor the minimal smoothed ||lr * grad||
            global_state["lr_grad_norm_avg_min"] = \
                np.exp(global_state["lr_grad_norm_avg"] / self.zero_debias_factor())
        else:
            global_state["lr_grad_norm_avg"] = global_state["lr_grad_norm_avg"] * beta \
                                               + (1 - beta) * np.log(
                self._lr * np.sqrt(global_state['grad_norm_squared']) + eps)
            global_state["lr_grad_norm_avg_min"] = \
                min(global_state["lr_grad_norm_avg_min"],
                    np.exp(global_state["lr_grad_norm_avg"] / self.zero_debias_factor()))

    def before_apply(self):
        # compute running average of gradient and norm of gradient
        beta = self._beta
        global_state = self._global_state
        if self._iter == 0:
            global_state["grad_norm_squared_avg"] = 0.0

        global_state["grad_norm_squared"] = 0.0
        for group_id, group in enumerate(self._optimizer.param_groups):
            for p_id, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_grad_norm_squared = torch.sum(grad * grad)
                global_state['grad_norm_squared'] += param_grad_norm_squared

                if self._verbose:
                    logging.debug("Iteration  %f", self._iter)
                    logging.debug("param grad squared gid %d, pid %d, %f, log scale: %f", group_id, p_id,
                                  param_grad_norm_squared,
                                  np.log(param_grad_norm_squared + 1e-10) / np.log(10))

        if self._iter >= 1:
            self._exploding_grad_clip_thresh = self._h_max
            self._exploding_grad_clip_target_value = np.sqrt(self._h_max)
            if global_state['grad_norm_squared'] >= self._exploding_grad_clip_thresh:
                self._exploding_grad_detected = True
            else:
                self._exploding_grad_detected = False

        global_state['grad_norm_squared_avg'] = \
            global_state['grad_norm_squared_avg'] * beta + (1 - beta) * global_state['grad_norm_squared']

        if self._verbose:
            logging.debug("overall grad norm squared %f, log scale: %f",
                          global_state['grad_norm_squared'],
                          np.log(global_state['grad_norm_squared'] + 1e-10) / np.log(10))

        if self._sparsity_debias:
            self.grad_sparsity()

        self.curvature_range()
        self.grad_variance()
        self.dist_to_opt()

        if self._verbose:
            logging.debug("h_max %f ", self._h_max)
            logging.debug("h_min %f ", self._h_min)
            logging.debug("dist %f ", self._dist_to_opt)
            logging.debug("var %f ", self._grad_var)

        if self._iter > 0:
            self.get_mu()
            self.get_lr()

            self._lr = beta * self._lr + (1 - beta) * self._lr_t
            self._mu = beta * self._mu + (1 - beta) * self._mu_t

            if self._verbose:
                logging.debug("lr_t %f", self._lr_t)
                logging.debug("mu_t %f", self._mu_t)
                logging.debug("lr %f", self._lr)
                logging.debug("mu %f", self._mu)
        return

    def get_lr(self):
        self._lr_t = (1.0 - math.sqrt(self._mu_t)) ** 2 / (self._h_min + eps)
        # slow start of lr to prevent huge lr when there is only a few iteration finished
        self._lr_t = min(self._lr_t, self._lr_t * (self._iter + 1) / float(10.0 * self._curv_win_width))
        return

    def get_cubic_root(self):
        # We have the equation x^2 D^2 + (1-x)^4 * C / h_min^2
        # where x = sqrt(mu).
        # We substitute x, which is sqrt(mu), with x = y + 1.
        # It gives y^3 + py = q
        # where p = (D^2 h_min^2)/(2*C) and q = -p.
        # We use the Vieta's substution to compute the root.
        # There is only one real solution y (which is in [0, 1] ).
        # http://mathworld.wolfram.com/VietasSubstitution.html
        # eps in the numerator is to prevent momentum = 1 in case of zero gradient
        if np.isnan(self._dist_to_opt) or np.isnan(self._h_min) or np.isnan(self._grad_var) \
                or np.isinf(self._dist_to_opt) or np.isinf(self._h_min) or np.isinf(self._grad_var):
            logging.warning("Input to cubic solver has invalid nan/inf value!")
            raise Exception("Input to cubic solver has invalid nan/inf value!")

        p = (self._dist_to_opt + eps) ** 2 * (self._h_min + eps) ** 2 / 2 / (self._grad_var + eps)
        w3 = (-math.sqrt(p ** 2 + 4.0 / 27.0 * p ** 3) - p) / 2.0
        w = math.copysign(1.0, w3) * math.pow(math.fabs(w3), 1.0 / 3.0)
        y = w - p / 3.0 / (w + eps)
        x = y + 1

        if self._verbose:
            logging.debug("p %f, denominator %f", p, self._grad_var + eps)
            logging.debug("w3 %f ", w3)
            logging.debug("y %f, denominator %f", y, w + eps)

        if np.isnan(x) or np.isinf(x):
            logging.warning("Output from cubic is invalid nan/inf value!")
            raise Exception("Output from cubic is invalid nan/inf value!")

        return x

    def get_mu(self):
        root = self.get_cubic_root()
        dr = max((self._h_max + eps) / (self._h_min + eps), 1.0 + eps)
        self._mu_t = max(root ** 2, ((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2)
        return

    def update_hyper_param(self):
        for group in self._optimizer.param_groups:
            group['momentum'] = self._mu_t
            # group['momentum'] = max(self._mu, self._mu_t)
            if self._force_non_inc_step == False:
                group['lr'] = self._lr_t * self._lr_factor
                # a loose clamping to prevent catastrophically large move. If the move
                # is too large, we set lr to 0 and only use the momentum to move
                if self._adapt_clip and (group['lr'] * np.sqrt(
                        self._global_state['grad_norm_squared']) >= self._catastrophic_move_thresh):
                    group['lr'] = self._catastrophic_move_thresh / np.sqrt(
                        self._global_state['grad_norm_squared'] + eps)
                    if self._verbose:
                        logging.warning("clip catastropic move!")
            elif self._iter > self._curv_win_width:
                # force to guarantee lr * grad_norm not increasing dramatically.
                # Not necessary for basic use. Please refer to the comments
                # in YFOptimizer.__init__ for more details
                self.lr_grad_norm_avg()
                debias_factor = self.zero_debias_factor()
                group['lr'] = min(self._lr * self._lr_factor,
                                  2.0 * self._global_state["lr_grad_norm_avg_min"] \
                                  / (np.sqrt(
                                      np.exp(self._global_state['grad_norm_squared_avg_log'] / debias_factor)) + eps))
        return

    def auto_clip_thresh(self):
        # Heuristic to automatically prevent sudden exploding gradient
        # Not necessary for basic use.
        return math.sqrt(self._h_max) * self._auto_clip_fac

    def step(self):
        # add weight decay
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

        if self._clip_thresh != None:
            torch.nn.utils.clip_grad_norm(self._var_list, self._clip_thresh)
        elif self._iter != 0 and self._auto_clip_fac != None:
            # do not clip the first iteration
            torch.nn.utils.clip_grad_norm(self._var_list, self.auto_clip_thresh())

        # loose threshold for preventing exploding gradients from destroying statistics
        if self._adapt_clip and (self._iter > 1):
            torch.nn.utils.clip_grad_norm(self._var_list, np.sqrt(self._stat_protect_fac * self._h_max) + eps)

        try:
            # before appply
            self.before_apply()

            # update learning rate and momentum
            self.update_hyper_param()

            # periodically save model and states
            if self._iter % self._checkpoint_interval == 0:
                if self._use_disk_checkpoint and os.path.exists(self._checkpoint_dir):
                    checkpoint_path = self._checkpoint_dir + "/" + self._checkpoint_file
                    with open(checkpoint_path, "wb") as f:
                        cp.dump(self.state_dict(), f, protocol=2)
                else:
                    self._state_checkpoint = copy.deepcopy(self.state_dict())

            # protection from exploding gradient
            if self._exploding_grad_detected and self._verbose:
                logging.warning(
                    "exploding gradient detected: grad norm detection thresh %f , grad norm %f, grad norm after clip%f",
                    np.sqrt(self._exploding_grad_clip_thresh),
                    np.sqrt(self._global_state['grad_norm_squared']),
                    self._exploding_grad_clip_target_value)
            if self._adapt_clip and self._exploding_grad_detected:
                # print("exploding gradient detected: grad norm detection thresh ",
                # np.sqrt(self._exploding_grad_clip_thresh),
                #   "grad norm", np.sqrt(self._global_state['grad_norm_squared'] ),
                #   "grad norm after clip ", self._exploding_grad_clip_target_value)
                torch.nn.utils.clip_grad_norm(self._var_list, self._exploding_grad_clip_target_value + eps)

            self._optimizer.step()

            self._iter += 1
        except:
            # load the last checkpoint
            logging.warning("Numerical issue triggered restore with backup. Resuming from last checkpoint.")
            if self._use_disk_checkpoint and os.path.exists(self._checkpoint_dir):
                checkpoint_path = self._checkpoint_dir + "/" + self._checkpoint_file
                with open(checkpoint_path, "rb") as f:
                    self.load_state_dict_perturb(cp.load(f))
            else:
                self.load_state_dict_perturb(copy.deepcopy(self._state_checkpoint))

        return
